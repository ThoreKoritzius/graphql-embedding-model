from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.synthetic import expand_queries
from graphql_finetuning_pipeline.utils.embeddings import light_embed
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_json, write_jsonl


@dataclass
class DatasetBuildConfig:
    version: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42
    semantic_dedupe_threshold: float = 0.93
    leakage_threshold: float = 0.5
    min_quality_score: float = 0.6
    target_train_min: int = 20_000
    target_train_max: int = 50_000


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def semantic_dedupe(rows: list[QueryRecord], threshold: float = 0.93) -> list[QueryRecord]:
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


def leakage_filter(rows: list[QueryRecord], type_names: set[str], threshold: float = 0.5) -> list[QueryRecord]:
    by_family: dict[str, list[QueryRecord]] = defaultdict(list)
    for row in rows:
        by_family[row.family_id].append(row)

    out: list[QueryRecord] = []
    for family, frows in by_family.items():
        leaked = 0
        for row in frows:
            ql = row.query.lower()
            if any(t.lower() in ql for t in type_names):
                leaked += 1
        ratio = leaked / max(len(frows), 1)
        if ratio <= threshold:
            out.extend(frows)
    return out


def ambiguity_filter(rows: list[QueryRecord]) -> list[QueryRecord]:
    banned = {"whatever", "something", "anything", "stuff", "etc"}
    out = []
    for row in rows:
        toks = set(_normalize_query(row.query).split())
        if len(toks) < 3:
            continue
        if toks & banned:
            continue
        out.append(row)
    return out


def enforce_intent_balance(rows: list[QueryRecord], min_per_intent: int = 100) -> list[QueryRecord]:
    by_intent: dict[str, list[QueryRecord]] = defaultdict(list)
    for row in rows:
        by_intent[row.intent or "unknown"].append(row)

    unknown = by_intent.pop("unknown", [])
    out: list[QueryRecord] = []
    for _, vals in by_intent.items():
        vals = sorted(vals, key=lambda x: x.quality_score, reverse=True)
        out.extend(vals[: max(min_per_intent, len(vals))])
    out.extend(unknown)
    return out


def split_by_family(rows: list[QueryRecord], train_ratio: float, val_ratio: float, seed: int) -> list[QueryRecord]:
    rng = random.Random(seed)
    fams: dict[str, list[QueryRecord]] = defaultdict(list)
    for row in rows:
        fams[row.family_id].append(row)

    family_ids = list(fams)
    rng.shuffle(family_ids)
    n = len(family_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(family_ids[:n_train])
    val_ids = set(family_ids[n_train : n_train + n_val])

    out: list[QueryRecord] = []
    for fid, frows in fams.items():
        split = "test"
        if fid in train_ids:
            split = "train"
        elif fid in val_ids:
            split = "val"
        out.extend([r.model_copy(update={"split": split}) for r in frows])
    return out


def build_benchmark_suites(split_rows: list[QueryRecord], corpus: list[CorpusRecord]) -> dict[str, list[QueryRecord]]:
    synthetic_holdout = [r for r in split_rows if r.split == "test"]

    # Realism eval: prefer hard+medium rows from val/test.
    realism_eval = [
        r.model_copy(update={"source": "realism_eval"})
        for r in split_rows
        if r.split in {"val", "test"} and (r.difficulty in {"medium", "hard"})
    ]

    # Adversarial: create confusion-oriented rewrites.
    by_type = {c.type_id: c for c in corpus}
    corpus_ids = [c.type_id for c in corpus]
    adversarial: list[QueryRecord] = []
    for i, row in enumerate(synthetic_holdout):
        other = row.confuser_type_ids[0] if row.confuser_type_ids else corpus_ids[(i + 1) % max(len(corpus_ids), 1)]
        t_name = by_type.get(row.target_type_id).type_name if row.target_type_id in by_type else row.target_type_id
        o_name = by_type.get(other).type_name if other in by_type else other
        adversarial.append(
            row.model_copy(
                update={
                    "query_id": f"adv-{row.query_id}",
                    "query": f"I need results related to {t_name} but not {o_name}; which GraphQL type should I use?",
                    "source": "adversarial_eval",
                    "difficulty": "hard",
                }
            )
        )

    return {
        "synthetic_holdout": synthetic_holdout,
        "realism_eval": realism_eval,
        "adversarial_eval": adversarial,
    }


def build_dataset(
    openai_seed_rows: list[QueryRecord],
    corpus: list[CorpusRecord],
    out_dir: Path,
    schema_hash: str,
    cfg: DatasetBuildConfig,
    generation_config: dict,
) -> dict:
    dataset_dir = out_dir / "datasets" / f"v{cfg.version}"
    ensure_dir(dataset_dir)
    ensure_dir(dataset_dir / "benchmarks")

    print("Building dataset: expansion")
    expanded = expand_queries(openai_seed_rows, max_expansions_per_seed=12)
    print("Building dataset: quality filtering")
    rows = [r for r in expanded if r.quality_score >= cfg.min_quality_score]
    rows = ambiguity_filter(rows)
    rows = leakage_filter(rows, type_names={c.type_name for c in corpus}, threshold=cfg.leakage_threshold)
    rows = semantic_dedupe(rows, threshold=cfg.semantic_dedupe_threshold)
    print("Building dataset: intent balancing and split")
    rows = enforce_intent_balance(rows, min_per_intent=300)

    split_rows = split_by_family(rows, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio, seed=cfg.seed)

    # Scale target by up/down sampling of train split families.
    train_rows = [r for r in split_rows if r.split == "train"]
    if len(train_rows) < cfg.target_train_min and train_rows:
        extra = []
        while len(train_rows) + len(extra) < cfg.target_train_min:
            for r in train_rows:
                extra.append(r.model_copy(update={"query_id": f"{r.query_id}-ups{len(extra)}"}))
                if len(train_rows) + len(extra) >= cfg.target_train_min:
                    break
        split_rows.extend(extra)
    elif len(train_rows) > cfg.target_train_max:
        keep_ids = {r.query_id for r in sorted(train_rows, key=lambda x: x.quality_score, reverse=True)[: cfg.target_train_max]}
        split_rows = [r for r in split_rows if r.split != "train" or r.query_id in keep_ids]

    for split in ("train", "val", "test"):
        write_jsonl(dataset_dir / f"{split}.jsonl", [r.model_dump() for r in split_rows if r.split == split])

    benches = build_benchmark_suites(split_rows, corpus)
    for name, rows_ in benches.items():
        write_jsonl(dataset_dir / "benchmarks" / f"{name}.jsonl", [r.model_dump() for r in rows_])

    counts = Counter([r.split for r in split_rows])
    intent_counts = Counter([r.intent or "unknown" for r in split_rows])
    manifest = {
        "version": cfg.version,
        "schema_hash": schema_hash,
        "generation_config": generation_config,
        "counts": dict(counts),
        "intent_counts": dict(intent_counts),
        "seed_rows": len(openai_seed_rows),
        "post_filter_rows": len(split_rows),
        "dataset_dir": str(dataset_dir.resolve()),
    }
    write_json(dataset_dir / "manifest.json", manifest)
    return manifest
