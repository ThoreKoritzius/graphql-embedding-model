from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.structural_views import ensure_view_available, get_view_text, normalize_primary_retrieval_view
from graphql_finetuning_pipeline.eval.metrics import (
    aggregate,
    coverage_at_k,
    mrr_at_k,
    ndcg_at_k,
    pair_recall_at_k,
    recall_at_k,
    set_recall_at_k,
)
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution
from graphql_finetuning_pipeline.utils.io import ensure_dir, read_jsonl


def _slice_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[f"intent:{r.get('intent') or 'unknown'}"].append(r)
        buckets[f"difficulty:{r.get('difficulty') or 'unknown'}"].append(r)
        buckets[f"source:{r.get('source') or 'unknown'}"].append(r)

    out: dict[str, dict[str, float]] = {}
    for key, vals in buckets.items():
        out[key] = {
            "count": float(len(vals)),
            "recall@5": aggregate([v["recall@5"] for v in vals]),
            "mrr@10": aggregate([v["mrr@10"] for v in vals]),
            "ndcg@10": aggregate([v["ndcg@10"] for v in vals]),
        }
    return out


def _encode(model_or_ref: Any, texts: list[str]) -> np.ndarray:
    if isinstance(model_or_ref, str):
        return encode_with_resolution(model_or_ref, texts, allow_remote_fallback=True)
    if hasattr(model_or_ref, "encode"):
        emb = model_or_ref.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
    raise TypeError("model_or_ref must be a model reference string or an object with .encode")


def evaluate_benchmark_set(
    rows: list[QueryRecord],
    corpus_rows: list[CorpusRecord],
    model_ref: Any,
    *,
    top_k: int = 10,
    retrieval_view: str = "sdl",
) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
            "set_recall_any@5": 0.0,
            "set_recall_all@10": 0.0,
            "coverage@10": 0.0,
            "pair_recall@10": 0.0,
            "embedding_latency_seconds": {"corpus": 0.0, "queries": 0.0},
            "slice_metrics": {},
            "transfer_gap": {"seen_vs_unseen_recall@5_gap": 0.0},
            "per_query": [],
        }

    view = normalize_primary_retrieval_view(retrieval_view)
    ensure_view_available(corpus_rows, view)
    corpus_ids = [c.type_id for c in corpus_rows]
    corpus_texts = [get_view_text(c, view) for c in corpus_rows]

    t0 = time.perf_counter()
    corpus_emb = _encode(model_ref, corpus_texts)
    t1 = time.perf_counter()
    query_emb = _encode(model_ref, [q.query for q in rows])
    t2 = time.perf_counter()

    sims = cosine_similarity(query_emb, corpus_emb)

    per_query: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        ranked_idx = np.argsort(-sims[i])
        ranked_ids = [corpus_ids[j] for j in ranked_idx[:top_k]]
        per_query.append(
            {
                "query_id": row.query_id,
                "query": row.query,
                "target_type_id": row.target_type_id,
                "intent": row.intent,
                "difficulty": row.difficulty,
                "source": row.source,
                "top_k": ranked_ids,
                "recall@1": recall_at_k(ranked_ids, row.target_type_id, 1),
                "recall@5": recall_at_k(ranked_ids, row.target_type_id, 5),
                "recall@10": recall_at_k(ranked_ids, row.target_type_id, 10),
                "mrr@10": mrr_at_k(ranked_ids, row.target_type_id, 10),
                "ndcg@10": ndcg_at_k(ranked_ids, row.target_type_id, 10),
                "set_recall_any@5": set_recall_at_k(
                    ranked_ids, row.relevant_type_ids or [row.primary_type_id or row.target_type_id], 5, mode="any"
                ),
                "set_recall_all@10": set_recall_at_k(
                    ranked_ids, row.relevant_type_ids or [row.primary_type_id or row.target_type_id], 10, mode="all"
                ),
                "coverage@10": coverage_at_k(
                    ranked_ids, row.relevant_type_ids or [row.primary_type_id or row.target_type_id], 10
                ),
                "pair_recall@10": pair_recall_at_k(
                    ranked_ids,
                    (row.relation_pair or {}).get("primary", ""),
                    (row.relation_pair or {}).get("bridge", ""),
                    10,
                ),
                "world_split": row.world_split or "unknown",
            }
        )

    seen = [x for x in per_query if x.get("world_split") == "train"]
    unseen = [x for x in per_query if x.get("world_split") in {"val", "test"}]
    seen_recall5 = aggregate([x["recall@5"] for x in seen]) if seen else 0.0
    unseen_recall5 = aggregate([x["recall@5"] for x in unseen]) if unseen else 0.0

    summary = {
        "count": len(rows),
        "recall@1": aggregate([x["recall@1"] for x in per_query]),
        "recall@5": aggregate([x["recall@5"] for x in per_query]),
        "recall@10": aggregate([x["recall@10"] for x in per_query]),
        "mrr@10": aggregate([x["mrr@10"] for x in per_query]),
        "ndcg@10": aggregate([x["ndcg@10"] for x in per_query]),
        "set_recall_any@5": aggregate([x["set_recall_any@5"] for x in per_query]),
        "set_recall_all@10": aggregate([x["set_recall_all@10"] for x in per_query]),
        "coverage@10": aggregate([x["coverage@10"] for x in per_query]),
        "pair_recall@10": aggregate([x["pair_recall@10"] for x in per_query]),
        "embedding_latency_seconds": {
            "corpus": t1 - t0,
            "queries": t2 - t1,
        },
        "slice_metrics": _slice_metrics(per_query),
        "transfer_gap": {"seen_vs_unseen_recall@5_gap": seen_recall5 - unseen_recall5},
        "per_query": per_query,
    }
    return summary


def run_benchmarks(
    benchmark_dir: Path,
    corpus_rows: list[CorpusRecord],
    model_ref: str,
    out_dir: Path,
    retrieval_view: str = "sdl",
) -> dict[str, Any]:
    ensure_dir(out_dir)
    results: dict[str, Any] = {}

    for f in sorted(benchmark_dir.glob("*.jsonl")):
        name = f.stem
        rows = [QueryRecord.model_validate(x) for x in read_jsonl(f)]
        results[name] = evaluate_benchmark_set(rows, corpus_rows, model_ref, retrieval_view=retrieval_view)
        (out_dir / f"{name}_summary.json").write_text(json.dumps(results[name], indent=2), encoding="utf-8")

    (out_dir / "benchmarks_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
