from __future__ import annotations

import random
import re
from collections import defaultdict

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord

INTENTS = {
    "lookup": [
        "what is {type}",
        "show me details of {type}",
        "explain graphql type for {type}",
    ],
    "capability": [
        "what can i query about {topic}",
        "which graphql type handles {topic}",
        "where do i fetch {topic}",
    ],
    "filtering": [
        "how can i filter {topic}",
        "which type supports filtering {topic}",
    ],
    "aggregation": [
        "how do i aggregate {topic}",
        "which type gives summary for {topic}",
    ],
    "troubleshooting": [
        "query for {topic} returns null, which type should i use",
        "i cannot find {topic} in schema, where is it",
    ],
    "comparative": [
        "difference between {type} and {other}",
        "when should i use {type} instead of {other}",
    ],
}

VERB_SWAPS = {
    "show": "list",
    "find": "locate",
    "fetch": "retrieve",
    "explain": "describe",
    "difference": "compare",
}

ABBREVIATIONS = {
    "information": "info",
    "identifier": "id",
    "number": "num",
    "query": "qry",
}


def _topic(c: CorpusRecord) -> str:
    if c.metadata.get("field_count", 0) > 0:
        return c.type_name.lower()
    return c.type_name.replace("Type", "").lower()


def bootstrap_queries(corpus: list[CorpusRecord], rng_seed: int = 7) -> list[QueryRecord]:
    rng = random.Random(rng_seed)
    queries: list[QueryRecord] = []

    for c in corpus:
        neighbors = [x for x in corpus if x.type_id != c.type_id]
        confuser = rng.choice(neighbors).type_name if neighbors else c.type_name

        family_counter = 0
        for intent, templates in INTENTS.items():
            for template in templates:
                q = template.format(type=c.type_name, topic=_topic(c), other=confuser)
                query_id = f"seed-{c.type_id}-{intent}-{family_counter}"
                family_id = f"{c.type_id}-{intent}-{family_counter}"
                queries.append(
                    QueryRecord(
                        query_id=query_id,
                        query=q,
                        target_type_id=c.type_id,
                        split="train",
                        source="bootstrap",
                        family_id=family_id,
                        quality_score=0.8,
                    )
                )
                family_counter += 1
    return queries


def _rewrite(query: str) -> set[str]:
    out = {query}
    for src, dst in VERB_SWAPS.items():
        out.add(re.sub(rf"\b{re.escape(src)}\b", dst, query, flags=re.IGNORECASE))
    for src, dst in ABBREVIATIONS.items():
        out.add(re.sub(rf"\b{re.escape(src)}\b", dst, query, flags=re.IGNORECASE))
    out.add(query.replace("graphql", "gql"))
    out.add(query.replace("how do i", "how can i"))
    return {x.strip() for x in out if x.strip()}


def _inject_noise(query: str) -> set[str]:
    noisy = {query}
    noisy.add(query.lower())
    noisy.add(query.replace("?", ""))
    noisy.add(query.replace("  ", " "))
    noisy.add(query.replace("type", "typ"))
    return {x.strip() for x in noisy if x.strip()}


def expand_queries(seed_queries: list[QueryRecord], max_expansions_per_seed: int = 6) -> list[QueryRecord]:
    expanded: list[QueryRecord] = []
    for q in seed_queries:
        variants = set()
        for rewritten in _rewrite(q.query):
            variants.update(_inject_noise(rewritten))

        variants = list(variants)[: max_expansions_per_seed + 1]
        for i, variant in enumerate(variants):
            source = "bootstrap" if i == 0 else "deterministic-augment"
            score = 0.85 if source == "bootstrap" else 0.65
            expanded.append(
                QueryRecord(
                    query_id=f"{q.query_id}-v{i}",
                    query=variant,
                    target_type_id=q.target_type_id,
                    split="train",
                    source=source,
                    family_id=q.family_id,
                    quality_score=score,
                )
            )
    return expanded


def quality_filter(
    queries: list[QueryRecord],
    type_names: set[str],
    leakage_threshold: float = 0.7,
    min_score: float = 0.55,
) -> list[QueryRecord]:
    accepted = [q for q in queries if q.quality_score >= min_score and len(q.query.split()) >= 3]

    # Deduplicate near identical by normalization.
    dedup: dict[str, QueryRecord] = {}
    for q in accepted:
        norm = re.sub(r"\s+", " ", q.query.lower()).strip()
        if norm not in dedup or q.quality_score > dedup[norm].quality_score:
            dedup[norm] = q

    accepted = list(dedup.values())

    # Leakage guard: if query contains exact target type name too often in a family, drop those families.
    by_family: dict[str, list[QueryRecord]] = defaultdict(list)
    for q in accepted:
        by_family[q.family_id].append(q)

    final: list[QueryRecord] = []
    for _, family_rows in by_family.items():
        leaked = 0
        for row in family_rows:
            row_low = row.query.lower()
            if any(tn.lower() in row_low for tn in type_names):
                leaked += 1
        if family_rows and (leaked / len(family_rows)) > leakage_threshold:
            continue
        final.extend(family_rows)

    return final


def split_queries(
    rows: list[QueryRecord], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42
) -> list[QueryRecord]:
    rng = random.Random(seed)
    fams: dict[str, list[QueryRecord]] = defaultdict(list)
    for row in rows:
        fams[row.family_id].append(row)

    family_ids = list(fams.keys())
    rng.shuffle(family_ids)
    n = len(family_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_fams = set(family_ids[:n_train])
    val_fams = set(family_ids[n_train : n_train + n_val])

    out: list[QueryRecord] = []
    for fid, frows in fams.items():
        split = "test"
        if fid in train_fams:
            split = "train"
        elif fid in val_fams:
            split = "val"

        for row in frows:
            out.append(row.model_copy(update={"split": split}))
    return out


def build_benchmark_sets(
    all_rows: list[QueryRecord], corpus: list[CorpusRecord], realism_max: int = 50
) -> dict[str, list[QueryRecord]]:
    synthetic_holdout = [r for r in all_rows if r.split == "test"]

    # Manual realism starter set to be curated by humans.
    realism_seed: list[QueryRecord] = []
    for r in [x for x in all_rows if x.split == "val"][:realism_max]:
        realism_seed.append(
            r.model_copy(
                update={
                    "source": "manual-realism-seed",
                    "quality_score": max(r.quality_score, 0.9),
                }
            )
        )

    # Adversarial set: force confusion between neighboring type names.
    by_type = {c.type_id: c for c in corpus}
    corpus_ids = [c.type_id for c in corpus]
    adversarial: list[QueryRecord] = []
    for i, row in enumerate(synthetic_holdout):
        other = corpus_ids[(i + 1) % len(corpus_ids)] if corpus_ids else row.target_type_id
        target_name = by_type.get(row.target_type_id).type_name if row.target_type_id in by_type else row.target_type_id
        other_name = by_type.get(other).type_name if other in by_type else other
        text = f"i need {target_name.lower()} info, not {other_name.lower()}, which graphql type is correct"
        adversarial.append(
            QueryRecord(
                query_id=f"adv-{row.query_id}",
                query=text,
                target_type_id=row.target_type_id,
                split="test",
                source="adversarial-ambiguity",
                family_id=f"adv-{row.family_id}",
                quality_score=0.8,
            )
        )

    return {
        "synthetic_holdout": synthetic_holdout,
        "realism_seed": realism_seed,
        "adversarial_ambiguity": adversarial,
    }
