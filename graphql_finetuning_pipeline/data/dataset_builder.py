from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import random

import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.structural_views import corpus_structural_hash
from graphql_finetuning_pipeline.utils.embeddings import light_embed
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_json, write_jsonl


_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|[_.]")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_TRIVIAL_LEAKAGE_TOKENS = {"id", "name", "type", "value", "data", "item", "list", "count", "total", "date", "time"}


def _light_stem(token: str) -> str:
    t = token.lower()
    for suffix in ("ies", "ing", "ers", "ier", "ied", "ies", "ed", "es", "er", "ly", "s"):
        if len(t) > len(suffix) + 2 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def _schema_terms_from(owner_type: str, field_name: str) -> set[str]:
    raw = list(_CAMEL_SPLIT_RE.split(owner_type)) + list(_CAMEL_SPLIT_RE.split(field_name))
    out: set[str] = set()
    for part in raw:
        for tok in _TOKEN_RE.findall((part or "").lower()):
            if len(tok) < 3 or tok in _TRIVIAL_LEAKAGE_TOKENS:
                continue
            out.add(_light_stem(tok))
    return out


def _query_stems(text: str) -> set[str]:
    return {_light_stem(tok) for tok in _TOKEN_RE.findall(text.lower()) if len(tok) >= 3}


@dataclass
class DatasetBuildConfig:
    version: int = 1
    seed: int = 42
    semantic_dedupe_threshold: float = 0.97
    leakage_threshold: float = 0.2
    min_quality_score: float = 0.25
    target_train_min: int = 10000
    target_train_max: int = 50000


def _normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


def semantic_dedupe(rows: list[QueryRecord], threshold: float = 0.9, max_per_family: int = 4) -> list[QueryRecord]:
    """Deduplicate queries in two passes.

    Pass 1 (family-level): within each (world_id, positive_coordinate) group
    keep at most ``max_per_family`` queries by cosine diversity on the light
    embedder. This collapses the 4-phrasing clusters GPT generates per field
    and prevents any single field from dominating the split.

    Pass 2 (global): standard greedy cosine dedupe across all surviving rows
    at ``threshold``. Removes near-duplicate phrasings that happen to target
    different fields in different worlds.
    """
    if not rows:
        return []

    # Pass 1: per-family cap
    from collections import defaultdict
    families: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in rows:
        key = f"{r.world_id or 'unknown'}:{r.positive_coordinate}"
        families[key].append(r)

    after_family: list[QueryRecord] = []
    for fam_rows in families.values():
        if len(fam_rows) <= max_per_family:
            after_family.extend(fam_rows)
            continue
        embs = light_embed([_normalize_query(r.query) for r in fam_rows])
        kept: list[int] = [0]
        kept_embs = [embs[0]]
        for i in range(1, len(fam_rows)):
            sims = cosine_similarity(np.asarray([embs[i]]), np.asarray(kept_embs))[0]
            if float(np.max(sims)) < threshold:
                kept.append(i)
                kept_embs.append(embs[i])
            if len(kept) >= max_per_family:
                break
        after_family.extend(fam_rows[j] for j in kept)

    # Pass 2: global dedup
    embs2 = light_embed([_normalize_query(r.query) for r in after_family])
    keep: list[QueryRecord] = []
    keep_embs: list[np.ndarray] = []
    for i, row in enumerate(tqdm(after_family, desc="Semantic dedupe", unit="query")):
        if not keep_embs:
            keep.append(row)
            keep_embs.append(embs2[i])
            continue
        sims = cosine_similarity(np.asarray([embs2[i]]), np.asarray(keep_embs))[0]
        if float(np.max(sims)) < threshold:
            keep.append(row)
            keep_embs.append(embs2[i])
    return keep


_LEAKAGE_TRIVIAL = {
    "id", "name", "type", "value", "data", "item", "list", "count", "total",
    "date", "time", "user", "post", "order", "cart", "hotel", "flight", "trip",
    "product", "catalog", "unit", "text", "body", "record", "mode", "state",
    "status", "created", "updated", "amount", "price", "active", "label",
    "title", "notes", "start", "email", "phone", "image", "score", "error",
    "token", "slug",
    # Common English words that appear as query verbs/nouns regardless of schema
    "field", "when", "last", "update", "create", "creation", "unique",
    "identifier", "display", "number", "model", "case", "times", "member",
    "which", "what", "this", "that", "with", "from", "have", "does", "used",
    "tell", "show", "give", "find", "look", "help", "need", "want", "know",
    "there", "here", "some", "only", "also", "more", "most", "many", "each",
    "been", "info", "read", "like", "make", "take", "come", "work", "year",
}


def leakage_filter(rows: list[QueryRecord], canonical_terms: set[str], threshold: float = 0.0, world_canonical: dict[str, set[str]] | None = None) -> list[QueryRecord]:
    """World-level leakage guard.

    A world is dropped if more than ``threshold`` of its queries contain any
    non-trivial canonical token from **that world's own schema** (not the
    global merged corpus). When ``world_canonical`` is provided it is used
    for per-world lookup; otherwise ``canonical_terms`` is used globally.

    Using per-world canonical prevents real schemas (e.g. GitHub's 6,901
    fields) from polluting the canonical set with common English words and
    nuking all synthetic worlds.
    """
    by_world: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in rows:
        by_world[r.world_id or "unknown"].append(r)
    out: list[QueryRecord] = []
    for world_id, vals in by_world.items():
        terms = world_canonical.get(world_id, canonical_terms) if world_canonical else canonical_terms
        meaningful = {t for t in terms if len(t) >= 4 and t not in _LEAKAGE_TRIVIAL}
        leaked = sum(1 for row in vals if any(term in row.query.lower() for term in meaningful))
        ratio = leaked / max(len(vals), 1)
        if ratio <= threshold:
            out.extend(vals)
    return out


def strict_leakage_filter(rows: list[QueryRecord]) -> list[QueryRecord]:
    """Row-level leakage guard using stemmed tokens.

    Drops a query when any stemmed token in the query matches a stemmed
    token from the positive coordinate's owner type or field name. This
    catches camelCase splits ("authorId" -> {"author", "identifier"}),
    plurals, and common suffix variants that the world-level
    ``leakage_filter`` will miss. Trivial scalar tokens like ``id`` or
    ``name`` are ignored so genuinely generic queries are not dropped.
    """
    out: list[QueryRecord] = []
    for row in rows:
        owner = row.owner_type or row.positive_coordinate.split(".", 1)[0]
        field = row.field_name or (row.positive_coordinate.split(".", 1)[1] if "." in row.positive_coordinate else "")
        schema_stems = _schema_terms_from(owner, field)
        if not schema_stems:
            out.append(row)
            continue
        if schema_stems & _query_stems(row.query):
            continue
        out.append(row)
    return out


def ambiguity_filter(rows: list[QueryRecord]) -> list[QueryRecord]:
    banned = {"something", "anything", "whatever", "some data", "etc"}
    out: list[QueryRecord] = []
    for row in rows:
        qn = _normalize_query(row.query)
        if len(qn.split()) < 4:
            continue
        if any(x in qn for x in banned):
            continue
        if not row.positive_coordinate:
            continue
        out.append(row)
    return out


def _name_stem(name: str) -> str:
    """Return the "meaningful" stem of a camelCase field name.

    ``status`` -> ``status``. ``statusHistory`` -> ``status`` (prefix).
    ``currentStatus`` -> ``status`` (suffix after split). Used to cluster
    fields that a retriever can easily confuse by lexical overlap.
    """
    # camelCase split
    parts: list[str] = []
    cur = ""
    for ch in name:
        if ch.isupper() and cur:
            parts.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        parts.append(cur)
    # Prefer the first part >= 4 chars; otherwise the longest part.
    candidates = [p.lower() for p in parts if len(p) >= 4]
    if candidates:
        return candidates[0]
    return max((p.lower() for p in parts), key=len, default=name.lower())


def _name_similarity_siblings(target: CorpusRecord, corpus_rows: list[CorpusRecord]) -> list[str]:
    """Coordinates whose field-name shares a stem with the target's field-name.

    This is the cluster that traditional embedders most often fumble:
    ``Order.status`` vs ``Order.statusHistory`` vs ``Order.currentStatus``.
    We compute it on the *field* name (not the owner) so same-owner siblings
    and cross-owner siblings both surface.
    """
    stem = _name_stem(target.field_name)
    if len(stem) < 4:
        return []
    siblings: list[str] = []
    for c in corpus_rows:
        if c.coordinate == target.coordinate:
            continue
        if _name_stem(c.field_name) == stem or stem in c.field_name.lower() or _name_stem(c.field_name) in target.field_name.lower():
            siblings.append(c.coordinate)
    return siblings


def _mine_negative_coordinates(row: QueryRecord, corpus_by_coord: dict[str, CorpusRecord], corpus_rows: list[CorpusRecord], rng: random.Random) -> tuple[list[str], list[str]]:
    target = corpus_by_coord.get(row.positive_coordinate)
    if target is None:
        return row.negative_coordinates[:8], row.confuser_tags

    # Restrict pools to the target's world so negatives are actually
    # retrievable candidates on that schema. Without this, models trained
    # with in-batch negatives learn trivial world-discrimination instead of
    # within-schema field discrimination.
    target_world = target.metadata.get("world_id")
    world_pool = [c for c in corpus_rows if c.metadata.get("world_id") == target_world] if target_world else corpus_rows

    same_owner = [c.coordinate for c in world_pool if c.owner_type == target.owner_type and c.coordinate != target.coordinate]
    same_field = [c.coordinate for c in world_pool if c.field_name == target.field_name and c.owner_type != target.owner_type]
    same_return = [c.coordinate for c in world_pool if c.return_type == target.return_type and c.coordinate != target.coordinate]
    semantic = [c.coordinate for c in world_pool if c.coordinate != target.coordinate and c.field_name in set(target.aliases or [])]
    name_siblings = _name_similarity_siblings(target, world_pool)

    negatives: list[str] = list(row.negative_coordinates)
    tags: list[str] = list(row.confuser_tags)
    # Order matters: the first negative tends to be the "hardest" in the
    # multi-negative-ranking loss because in-batch collision rates are
    # higher at low positions. Prioritize name-similarity siblings.
    for tag, pool in [
        ("name_similarity", name_siblings),
        ("structural", same_owner),
        ("lexical", same_field),
        ("argument-shape", same_return),
        ("semantic", semantic),
    ]:
        if pool:
            negatives.append(rng.choice(pool))
            tags.append(tag)
    if len(negatives) < 5:
        pool = [c.coordinate for c in world_pool if c.coordinate != target.coordinate and c.coordinate not in negatives]
        if pool:
            negatives.extend(rng.sample(pool, k=min(5 - len(negatives), len(pool))))
    return list(dict.fromkeys(negatives))[:8], list(dict.fromkeys(tags))


def _curated_queries_for_field(row: CorpusRecord, domain: str) -> list[tuple[str, str, str]]:
    hint = row.field_name.replace("Id", " identifier").replace("Cents", " amount").lower()
    templates = {
        "author": [
            ("Who wrote the record?", "authorship", "obvious"),
            ("I already have the object; which nested field tells me who owns it?", "root_vs_nested", "adversarial"),
        ],
        "status": [
            ("Which field tells me the current state?", "status", "obvious"),
            ("Not the history, the field with the current status", "status", "sibling-field"),
        ],
        "createdAt": [("When was this record created?", "temporal", "obvious")],
        "updatedAt": [("When was this record last changed?", "temporal", "obvious")],
        "email": [("What field stores the email address?", "lookup", "obvious")],
    }
    picked = templates.get(row.field_name, [(f"Which field gives me the {hint}?", "lookup", "obvious")])
    return [(q, intent, slice_name) for q, intent, slice_name in picked]


def build_benchmark_suites(split_rows: list[QueryRecord], corpus: list[CorpusRecord]) -> dict[str, list[QueryRecord]]:
    real_world_ids = {c.metadata.get("world_id") for c in corpus if c.metadata.get("source") == "real-schema"}
    test_rows = [r for r in split_rows if r.split == "test"]

    # realism_eval: the "looks-like-live-traffic" subset. Plain/ellipsis/
    # multi-clause rows without adversarial tagging. Leakage is NOT
    # re-applied here (it was applied upstream per-row); real users often
    # do mention partial schema tokens, so this benchmark is deliberately
    # closer to live conditions.
    realism_eval = [r for r in test_rows if not r.adversarial_tags]

    # adversarial_eval: the sibling-confuser / root-vs-nested slice where
    # a naive retriever is supposed to fail. This is the hardest slice.
    adversarial_eval = [r for r in test_rows if r.adversarial_tags]

    # ambiguity_eval: rows with ≥2 valid coordinates (path-to-root ambiguity).
    # Evaluated with coverage@k and set_recall@k, not strict top1.
    ambiguity_eval = [r for r in test_rows if len(r.relevant_coordinates or []) > 1]

    # real_schema_eval: test-set rows whose world is a real (ingested) SDL,
    # not a synthetic world. This is the honest transfer metric. Empty when
    # no real schemas were ingested.
    real_schema_eval = [r for r in test_rows if r.world_id in real_world_ids] if real_world_ids else []

    synthetic_holdout = test_rows

    curated_eval: list[QueryRecord] = []
    test_corpus = [c for c in corpus if c.metadata.get("world_id") and any(r.world_id == c.metadata.get("world_id") and r.split == "test" for r in split_rows)]
    if not test_corpus:
        test_corpus = list(corpus)
    seen: set[tuple[str, str]] = set()
    for c in test_corpus[:100]:
        for query, intent, slice_name in _curated_queries_for_field(c, c.metadata.get("domain", "unknown")):
            key = (query, c.coordinate)
            if key in seen:
                continue
            seen.add(key)
            curated_eval.append(
                QueryRecord(
                    query_id=f"curated-{c.doc_id}-{len(curated_eval)}",
                    query=query,
                    positive_coordinate=c.coordinate,
                    split="test",
                    source="curated-challenge",
                    family_id=f"curated:{c.coordinate}",
                    quality_score=1.0,
                    negative_coordinates=[],
                    relevant_coordinates=[c.coordinate],
                    owner_type=c.owner_type,
                    field_name=c.field_name,
                    world_id=c.metadata.get("world_id"),
                    domain=c.metadata.get("domain"),
                    world_split="test",
                    difficulty="hard" if slice_name in {"adversarial", "sibling-field"} else "medium",
                    intent=intent,
                    confuser_tags=[slice_name],
                    rationale_tags=["curated", "release-gate"],
                )
            )
            if len(curated_eval) >= 200:
                break
        if len(curated_eval) >= 200:
            break

    suites = {
        "realism_eval": realism_eval,
        "adversarial_eval": adversarial_eval,
        "synthetic_holdout": synthetic_holdout,
        "curated_challenge_eval": curated_eval,
        "ambiguity_eval": ambiguity_eval,
    }
    if real_schema_eval:
        suites["real_schema_eval"] = real_schema_eval
    return suites


def build_dataset(openai_seed_rows: list[QueryRecord], corpus: list[CorpusRecord], out_dir: Path, schema_hash: str, cfg: DatasetBuildConfig, generation_config: dict) -> dict:
    dataset_dir = out_dir / "datasets" / f"v{cfg.version}"
    ensure_dir(dataset_dir)
    ensure_dir(dataset_dir / "benchmarks")

    # Build per-world canonical so the leakage filter uses only each world's
    # own type/field names — not the global merged set (which includes real
    # schemas like GHES with 6,901 fields, polluting canonical with common words).
    world_canonical: dict[str, set[str]] = defaultdict(set)
    for c in corpus:
        wid = (c.metadata or {}).get("world_id", "unknown")
        world_canonical[wid].update({c.coordinate.lower(), c.owner_type.lower(), c.field_name.lower()})
    # Fallback global set for rows with unknown world_id.
    canonical_terms = {c.coordinate.lower() for c in corpus} | {c.owner_type.lower() for c in corpus} | {c.field_name.lower() for c in corpus}

    seed_by_split: dict[str, list[QueryRecord]] = {"train": [], "val": [], "test": []}
    for r in openai_seed_rows:
        split = (r.world_split or r.split or "train").lower()
        if split not in seed_by_split:
            split = "train"
        if r.quality_score >= cfg.min_quality_score:
            seed_by_split[split].append(r.model_copy(update={"split": split}))

    if not seed_by_split["val"] or not seed_by_split["test"]:
        raise ValueError("OpenAI seed rows are missing held-out splits. Expected non-empty val and test rows.")

    split_rows: list[QueryRecord] = []
    pre_filter_counts = {k: len(v) for k, v in seed_by_split.items()}
    post_filter_counts: dict[str, int] = {}
    stage_counts: dict[str, dict[str, int]] = {}
    rng = random.Random(cfg.seed)
    corpus_by_coord = {c.coordinate: c for c in corpus}

    for split, rows in seed_by_split.items():
        after_ambiguity = ambiguity_filter(rows)
        after_world_leakage = leakage_filter(after_ambiguity, canonical_terms=canonical_terms, threshold=cfg.leakage_threshold, world_canonical=world_canonical)
        after_strict_leakage = strict_leakage_filter(after_world_leakage) if split in {"val", "test"} else after_world_leakage
        deduped = semantic_dedupe(after_strict_leakage, threshold=cfg.semantic_dedupe_threshold)
        if split in {"val", "test"} and not deduped and after_strict_leakage:
            deduped = after_strict_leakage
        updated: list[QueryRecord] = []
        for row in deduped:
            negatives, tags = _mine_negative_coordinates(row, corpus_by_coord, corpus, rng)
            updated.append(row.model_copy(update={"negative_coordinates": negatives, "confuser_tags": tags, "relevant_coordinates": row.relevant_coordinates or [row.positive_coordinate]}))
        post_filter_counts[split] = len(updated)
        stage_counts[split] = {
            "pre_filter": len(rows),
            "after_ambiguity": len(after_ambiguity),
            "after_world_leakage": len(after_world_leakage),
            "after_strict_leakage": len(after_strict_leakage),
            "after_semantic_dedupe": len(deduped),
            "final": len(updated),
        }
        split_rows.extend([r.model_copy(update={"split": split}) for r in updated])

    if not [r for r in split_rows if r.split == "train"]:
        raise ValueError("No train rows remain after filtering.")

    for split in ("train", "val", "test"):
        write_jsonl(dataset_dir / f"{split}.jsonl", [r.model_dump() for r in split_rows if r.split == split])

    benches = build_benchmark_suites(split_rows, corpus)
    for name, rows_ in benches.items():
        write_jsonl(dataset_dir / "benchmarks" / f"{name}.jsonl", [r.model_dump() for r in rows_])

    source_counts = dict(Counter([r.source or "unknown" for r in split_rows]))
    intent_counts = dict(Counter([r.intent or "unknown" for r in split_rows]))
    sample_rng = random.Random(cfg.seed + 1)
    sample_rows = sample_rng.sample(split_rows, k=min(20, len(split_rows)))
    sanity_sample = [
        {
            "split": r.split,
            "query": r.query,
            "positive_coordinate": r.positive_coordinate,
            "intent": r.intent,
            "source": r.source,
            "quality_score": r.quality_score,
            "negatives": r.negative_coordinates[:4],
        }
        for r in sample_rows
    ]

    sanity_report = {
        "pre_filter_split_counts": pre_filter_counts,
        "per_stage_counts": stage_counts,
        "post_filter_split_counts": post_filter_counts,
        "source_counts": source_counts,
        "intent_counts": intent_counts,
        "fallback_share": round(
            sum(v for k, v in source_counts.items() if "fallback" in k) / max(sum(source_counts.values()), 1),
            4,
        ),
        "config": {
            "semantic_dedupe_threshold": cfg.semantic_dedupe_threshold,
            "leakage_threshold": cfg.leakage_threshold,
            "min_quality_score": cfg.min_quality_score,
        },
        "warnings": [],
    }
    if sanity_report["fallback_share"] >= 0.5:
        sanity_report["warnings"].append(
            f"{sanity_report['fallback_share']:.0%} of post-filter rows are from the local fallback. "
            "Expect noticeably lower model quality than an OpenAI-sourced run."
        )
    for split, counts in stage_counts.items():
        if counts["pre_filter"] > 0 and counts["final"] / counts["pre_filter"] < 0.2:
            sanity_report["warnings"].append(
                f"Aggressive filtering on split '{split}': {counts['final']}/{counts['pre_filter']} rows survived "
                "(<20%). Check semantic_dedupe_threshold and leakage_threshold."
            )

    manifest = {
        "version": cfg.version,
        "schema_hash": schema_hash,
        "corpus_view_version": 2,
        "corpus_structural_hash": corpus_structural_hash(corpus),
        "generation_config": generation_config,
        "counts": dict(Counter([r.split for r in split_rows])),
        "seed_split_counts": pre_filter_counts,
        "post_filter_split_counts": post_filter_counts,
        "per_stage_counts": stage_counts,
        "source_counts": source_counts,
        "intent_counts": intent_counts,
        "world_split_counts": dict(Counter([r.world_split or "unknown" for r in split_rows])),
        "domain_counts": dict(Counter([r.domain or "unknown" for r in split_rows])),
        "seed_rows": len(openai_seed_rows),
        "post_filter_rows": len(split_rows),
        "coverage_distribution": {
            "avg_relevant_coordinates": sum(len(r.relevant_coordinates) for r in split_rows) / max(len(split_rows), 1),
            "adversarial_ratio": sum(1 for r in split_rows if r.adversarial_tags) / max(len(split_rows), 1),
            "confuser_ratio": sum(1 for r in split_rows if r.confuser_tags) / max(len(split_rows), 1),
        },
        "dataset_dir": str(dataset_dir.resolve()),
        "release_gate_benchmark": "curated_challenge_eval",
        "sanity_warnings": sanity_report["warnings"],
    }
    write_json(dataset_dir / "manifest.json", manifest)
    write_json(dataset_dir / "sanity_report.json", sanity_report)
    write_jsonl(dataset_dir / "sanity_sample.jsonl", sanity_sample)
    write_jsonl(dataset_dir / "corpus.jsonl", [c.model_dump() for c in corpus])
    return manifest
