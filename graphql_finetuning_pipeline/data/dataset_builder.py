from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.structural_views import corpus_structural_hash
from graphql_finetuning_pipeline.utils.embeddings import light_embed
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_json, write_jsonl


@dataclass
class DatasetBuildConfig:
    version: int = 1
    seed: int = 42
    semantic_dedupe_threshold: float = 0.92
    leakage_threshold: float = 0.02
    min_quality_score: float = 0.65
    target_train_min: int = 200_000
    target_train_max: int = 500_000
    adversarial_ratio: float = 0.18
    noisy_ratio: float = 0.22


def _normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


def semantic_dedupe(rows: list[QueryRecord], threshold: float = 0.92) -> list[QueryRecord]:
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


def leakage_filter(rows: list[QueryRecord], canonical_names: set[str], threshold: float = 0.02) -> list[QueryRecord]:
    by_world: dict[str, list[QueryRecord]] = defaultdict(list)
    for r in rows:
        by_world[r.world_id or "unknown"].append(r)

    out: list[QueryRecord] = []
    for _, vals in by_world.items():
        leaked = 0
        for row in vals:
            ql = row.query.lower()
            if any(name.lower() in ql for name in canonical_names):
                leaked += 1
        ratio = leaked / max(len(vals), 1)
        if ratio <= threshold:
            out.extend(vals)
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
        if not row.relevant_type_ids:
            continue
        out.append(row)
    return out


def _augment_rows(rows: list[QueryRecord], target_size: int) -> list[QueryRecord]:
    if len(rows) >= target_size:
        return rows
    out = list(rows)
    i = 0
    # Diversity-preserving augmentation: perturb casing/punctuation and add small noise tags.
    while len(out) < target_size and rows:
        base = rows[i % len(rows)]
        q = base.query
        if i % 3 == 0:
            q = q.replace("?", "")
        elif i % 3 == 1:
            q = q.lower()
        else:
            q = q + " please"
        out.append(
            base.model_copy(
                update={
                    "query_id": f"{base.query_id}-aug{i}",
                    "query": q,
                    "noise_tags": list(set(base.noise_tags + ["synthetic_augment"])),
                }
            )
        )
        i += 1
    return out


def build_benchmark_suites(split_rows: list[QueryRecord]) -> dict[str, list[QueryRecord]]:
    realism_eval = [r for r in split_rows if r.split == "test" and not r.adversarial_tags]
    adversarial_eval = [r for r in split_rows if r.split == "test" and r.adversarial_tags]
    compositional_eval = [
        r
        for r in split_rows
        if r.split == "test" and len(r.relevant_type_ids) >= 3 and len((r.relation_pair or {}).keys()) > 0
    ]
    return {
        "realism_eval": realism_eval,
        "adversarial_eval": adversarial_eval,
        "compositional_eval": compositional_eval,
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

    print("Building dataset: quality filtering")
    canonical_names = {c.type_name for c in corpus}
    seed_by_split: dict[str, list[QueryRecord]] = {"train": [], "val": [], "test": []}
    for r in openai_seed_rows:
        split = (r.world_split or r.split or "train").lower()
        if split not in seed_by_split:
            split = "train"
        if r.quality_score >= cfg.min_quality_score:
            seed_by_split[split].append(r.model_copy(update={"split": split}))

    if not seed_by_split["val"] or not seed_by_split["test"]:
        raise ValueError(
            "OpenAI seed rows are missing held-out splits. "
            "Expected non-empty val and test rows in `seed_pairs_v1.jsonl`. "
            "Re-run `graphft generate-openai-seed` until completion."
        )

    split_rows: list[QueryRecord] = []
    pre_filter_counts = {k: len(v) for k, v in seed_by_split.items()}
    post_filter_counts: dict[str, int] = {}

    # Filter each split independently to avoid train rows deduping away val/test.
    for split, rows in seed_by_split.items():
        filtered = ambiguity_filter(rows)
        filtered = leakage_filter(filtered, canonical_names=canonical_names, threshold=cfg.leakage_threshold)
        deduped = semantic_dedupe(filtered, threshold=cfg.semantic_dedupe_threshold)

        # Keep holdouts non-empty for reliable eval; if dedupe is too aggressive, keep pre-dedupe rows.
        if split in {"val", "test"} and not deduped and filtered:
            deduped = filtered

        post_filter_counts[split] = len(deduped)
        split_rows.extend([r.model_copy(update={"split": split}) for r in deduped])

    # Scale training set to desired size with augmentation while keeping val/test untouched.
    train_rows = [r for r in split_rows if r.split == "train"]
    train_rows = _augment_rows(train_rows, target_size=min(cfg.target_train_max, max(cfg.target_train_min, len(train_rows))))
    other_rows = [r for r in split_rows if r.split != "train"]
    split_rows = train_rows + other_rows

    # Build negative_type_ids mix: structural neighbors + confusers from corpus metadata.
    by_id = {c.type_id: c for c in corpus}
    for i, row in enumerate(split_rows):
        negs = list(row.negative_type_ids)
        c = by_id.get(row.primary_type_id or row.target_type_id)
        if c:
            neighbors = c.metadata.get("neighbors", [])
            negs.extend([x for x in neighbors if x not in row.relevant_type_ids])
        # fallback random negatives
        if len(negs) < 5:
            pool = [c.type_id for c in corpus if c.type_id not in set(row.relevant_type_ids + [row.primary_type_id or row.target_type_id])]
            negs.extend(pool[: max(0, 5 - len(negs))])
        split_rows[i] = row.model_copy(update={"negative_type_ids": list(dict.fromkeys(negs))[:12]})

    for split in ("train", "val", "test"):
        write_jsonl(dataset_dir / f"{split}.jsonl", [r.model_dump() for r in split_rows if r.split == split])

    benches = build_benchmark_suites(split_rows)
    for name, rows_ in benches.items():
        write_jsonl(dataset_dir / "benchmarks" / f"{name}.jsonl", [r.model_dump() for r in rows_])

    world_splits = Counter([r.world_split or "unknown" for r in split_rows])
    domain_counts = Counter([r.domain or "unknown" for r in split_rows])
    manifest = {
        "version": cfg.version,
        "schema_hash": schema_hash,
        "corpus_view_version": 1,
        "corpus_structural_hash": corpus_structural_hash(corpus),
        "generation_config": generation_config,
        "counts": dict(Counter([r.split for r in split_rows])),
        "seed_split_counts": pre_filter_counts,
        "post_filter_split_counts": post_filter_counts,
        "world_split_counts": dict(world_splits),
        "domain_counts": dict(domain_counts),
        "seed_rows": len(openai_seed_rows),
        "post_filter_rows": len(split_rows),
        "coverage_distribution": {
            "avg_relevant_types": sum(len(r.relevant_type_ids) for r in split_rows) / max(len(split_rows), 1),
            "adversarial_ratio": sum(1 for r in split_rows if r.adversarial_tags) / max(len(split_rows), 1),
            "noise_ratio": sum(1 for r in split_rows if r.noise_tags) / max(len(split_rows), 1),
        },
        "dataset_dir": str(dataset_dir.resolve()),
    }
    write_json(dataset_dir / "manifest.json", manifest)

    # Save generated corpus aligned with dataset version.
    write_jsonl(dataset_dir / "corpus.jsonl", [c.model_dump() for c in corpus])
    return manifest
