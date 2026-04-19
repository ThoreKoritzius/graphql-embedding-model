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
    semantic_dedupe_threshold: float = 0.9
    leakage_threshold: float = 0.0
    min_quality_score: float = 0.7
    target_train_min: int = 10000
    target_train_max: int = 50000


def _normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


def semantic_dedupe(rows: list[QueryRecord], threshold: float = 0.9) -> list[QueryRecord]:
    if not rows:
        return []
    embs = light_embed([_normalize_query(r.query) for r in rows])
    keep: list[QueryRecord] = []
    keep_embs: list[np.ndarray] = []
    for i, row in enumerate(tqdm(rows, desc="Semantic dedupe", unit="query")):
        if not keep_embs:
            keep.append(row)
            keep_embs.append(embs[i])
            continue
        sims = cosine_similarity(np.asarray([embs[i]]), np.asarray(keep_embs))[0]
        if float(np.max(sims)) < threshold:
            keep.append(row)
            keep_embs.append(embs[i])
    return keep


def leakage_filter(rows: list[QueryRecord], canonical_terms: set[str], threshold: float = 0.0) -> list[QueryRecord]:
    by_world: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in rows:
        by_world[r.world_id or "unknown"].append(r)
    out: list[QueryRecord] = []
    for vals in by_world.values():
        leaked = sum(1 for row in vals if any(term in row.query.lower() for term in canonical_terms))
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


def _mine_negative_coordinates(row: QueryRecord, corpus_by_coord: dict[str, CorpusRecord], corpus_rows: list[CorpusRecord], rng: random.Random) -> tuple[list[str], list[str]]:
    target = corpus_by_coord.get(row.positive_coordinate)
    if target is None:
        return row.negative_coordinates[:8], row.confuser_tags

    same_owner = [c.coordinate for c in corpus_rows if c.owner_type == target.owner_type and c.coordinate != target.coordinate]
    same_field = [c.coordinate for c in corpus_rows if c.field_name == target.field_name and c.owner_type != target.owner_type]
    same_return = [c.coordinate for c in corpus_rows if c.return_type == target.return_type and c.coordinate != target.coordinate]
    semantic = [c.coordinate for c in corpus_rows if c.coordinate != target.coordinate and c.field_name in set(target.aliases)]

    negatives: list[str] = list(row.negative_coordinates)
    tags: list[str] = list(row.confuser_tags)
    for tag, pool in [
        ("structural", same_owner),
        ("lexical", same_field),
        ("argument-shape", same_return),
        ("semantic", semantic),
    ]:
        if pool:
            negatives.append(rng.choice(pool))
            tags.append(tag)
    if len(negatives) < 5:
        pool = [c.coordinate for c in corpus_rows if c.coordinate != target.coordinate and c.coordinate not in negatives]
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
    return [(f"[{domain}] {q}", intent, slice_name) for q, intent, slice_name in picked]


def build_benchmark_suites(split_rows: list[QueryRecord], corpus: list[CorpusRecord]) -> dict[str, list[QueryRecord]]:
    realism_eval = [r for r in split_rows if r.split == "test" and not r.adversarial_tags]
    adversarial_eval = [r for r in split_rows if r.split == "test" and r.adversarial_tags]
    synthetic_holdout = [r for r in split_rows if r.split == "test"]

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

    return {
        "realism_eval": realism_eval,
        "adversarial_eval": adversarial_eval,
        "synthetic_holdout": synthetic_holdout,
        "curated_challenge_eval": curated_eval,
    }


def build_dataset(openai_seed_rows: list[QueryRecord], corpus: list[CorpusRecord], out_dir: Path, schema_hash: str, cfg: DatasetBuildConfig, generation_config: dict) -> dict:
    dataset_dir = out_dir / "datasets" / f"v{cfg.version}"
    ensure_dir(dataset_dir)
    ensure_dir(dataset_dir / "benchmarks")

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
    rng = random.Random(cfg.seed)
    corpus_by_coord = {c.coordinate: c for c in corpus}

    for split, rows in seed_by_split.items():
        filtered = ambiguity_filter(rows)
        filtered = leakage_filter(filtered, canonical_terms=canonical_terms, threshold=cfg.leakage_threshold)
        if split in {"val", "test"}:
            filtered = strict_leakage_filter(filtered)
        deduped = semantic_dedupe(filtered, threshold=cfg.semantic_dedupe_threshold)
        if split in {"val", "test"} and not deduped and filtered:
            deduped = filtered
        updated: list[QueryRecord] = []
        for row in deduped:
            negatives, tags = _mine_negative_coordinates(row, corpus_by_coord, corpus, rng)
            updated.append(row.model_copy(update={"negative_coordinates": negatives, "confuser_tags": tags, "relevant_coordinates": row.relevant_coordinates or [row.positive_coordinate]}))
        post_filter_counts[split] = len(updated)
        split_rows.extend([r.model_copy(update={"split": split}) for r in updated])

    if not [r for r in split_rows if r.split == "train"]:
        raise ValueError("No train rows remain after filtering.")

    for split in ("train", "val", "test"):
        write_jsonl(dataset_dir / f"{split}.jsonl", [r.model_dump() for r in split_rows if r.split == split])

    benches = build_benchmark_suites(split_rows, corpus)
    for name, rows_ in benches.items():
        write_jsonl(dataset_dir / "benchmarks" / f"{name}.jsonl", [r.model_dump() for r in rows_])

    manifest = {
        "version": cfg.version,
        "schema_hash": schema_hash,
        "corpus_view_version": 2,
        "corpus_structural_hash": corpus_structural_hash(corpus),
        "generation_config": generation_config,
        "counts": dict(Counter([r.split for r in split_rows])),
        "seed_split_counts": pre_filter_counts,
        "post_filter_split_counts": post_filter_counts,
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
    }
    write_json(dataset_dir / "manifest.json", manifest)
    write_jsonl(dataset_dir / "corpus.jsonl", [c.model_dump() for c in corpus])
    return manifest
