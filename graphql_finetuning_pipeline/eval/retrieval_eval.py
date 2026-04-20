from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.structural_views import ensure_view_available, get_view_text, normalize_primary_retrieval_view
from graphql_finetuning_pipeline.eval.metrics import aggregate, coverage_at_k, mrr_at_k, ndcg_at_k, recall_at_k, set_recall_at_k
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution
from graphql_finetuning_pipeline.utils.io import ensure_dir


def _same_owner_wrong_field_at_1(top_id: str, row: QueryRecord) -> float:
    if not top_id or top_id == row.positive_coordinate:
        return 0.0
    top_owner = top_id.split(".", 1)[0]
    return 1.0 if row.owner_type and top_owner == row.owner_type else 0.0


def evaluate(eval_rows: list[QueryRecord], corpus_rows: list[CorpusRecord], model_path_or_name: str, out_path: Path | None = None, retrieval_view: str = "semantic") -> dict:
    if not eval_rows:
        metrics = {
            "count": 0,
            "exact_match@1": 0.0,
            "recall@1": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "set_recall@5": 0.0,
            "set_recall@10": 0.0,
            "coverage@5": 0.0,
            "coverage@10": 0.0,
            "mrr@10": 0.0,
            "ndcg@10": 0.0,
            "same_owner_wrong_field_rate@1": 0.0,
            "ambiguity_rate": 0.0,
            "embedding_latency_seconds": {"corpus": 0.0, "queries": 0.0},
        }
        if out_path:
            ensure_dir(out_path.parent)
            out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    view = normalize_primary_retrieval_view(retrieval_view)
    ensure_view_available(corpus_rows, view)
    corpus_ids = [c.coordinate for c in corpus_rows]
    corpus_texts = [get_view_text(c, view) for c in corpus_rows]

    t0 = time.perf_counter()
    corpus_emb = encode_with_resolution(model_path_or_name, corpus_texts, allow_remote_fallback=True, prompt_name="document")
    t1 = time.perf_counter()
    query_emb = encode_with_resolution(model_path_or_name, [q.query for q in eval_rows], allow_remote_fallback=True, prompt_name="query")
    t2 = time.perf_counter()

    sims = cosine_similarity(query_emb, corpus_emb)
    metrics = {k: [] for k in ["exact", "r1", "r3", "r5", "r10", "sr5", "sr10", "cov5", "cov10", "mrr10", "ndcg10", "owner_wrong", "ambig"]}
    for i, row in enumerate(eval_rows):
        ranked_idx = np.argsort(-sims[i])
        ranked_ids = [corpus_ids[j] for j in ranked_idx]
        relevant = [r for r in dict.fromkeys([row.positive_coordinate] + list(row.relevant_coordinates or [])) if r]
        top_id = ranked_ids[0] if ranked_ids else ""
        metrics["exact"].append(1.0 if top_id in relevant else 0.0)
        metrics["r1"].append(recall_at_k(ranked_ids, row.positive_coordinate, 1))
        metrics["r3"].append(recall_at_k(ranked_ids, row.positive_coordinate, 3))
        metrics["r5"].append(recall_at_k(ranked_ids, row.positive_coordinate, 5))
        metrics["r10"].append(recall_at_k(ranked_ids, row.positive_coordinate, 10))
        metrics["sr5"].append(set_recall_at_k(ranked_ids, relevant, 5, mode="any"))
        metrics["sr10"].append(set_recall_at_k(ranked_ids, relevant, 10, mode="any"))
        metrics["cov5"].append(coverage_at_k(ranked_ids, relevant, 5))
        metrics["cov10"].append(coverage_at_k(ranked_ids, relevant, 10))
        metrics["mrr10"].append(mrr_at_k(ranked_ids, row.positive_coordinate, 10))
        metrics["ndcg10"].append(ndcg_at_k(ranked_ids, row.positive_coordinate, 10))
        metrics["owner_wrong"].append(_same_owner_wrong_field_at_1(top_id, row))
        metrics["ambig"].append(1.0 if len(relevant) > 1 else 0.0)

    summary = {
        "count": len(eval_rows),
        "exact_match@1": aggregate(metrics["exact"]),
        "recall@1": aggregate(metrics["r1"]),
        "recall@3": aggregate(metrics["r3"]),
        "recall@5": aggregate(metrics["r5"]),
        "recall@10": aggregate(metrics["r10"]),
        "set_recall@5": aggregate(metrics["sr5"]),
        "set_recall@10": aggregate(metrics["sr10"]),
        "coverage@5": aggregate(metrics["cov5"]),
        "coverage@10": aggregate(metrics["cov10"]),
        "mrr@10": aggregate(metrics["mrr10"]),
        "ndcg@10": aggregate(metrics["ndcg10"]),
        "same_owner_wrong_field_rate@1": aggregate(metrics["owner_wrong"]),
        "ambiguity_rate": aggregate(metrics["ambig"]),
        "embedding_latency_seconds": {"corpus": t1 - t0, "queries": t2 - t1},
    }
    if out_path:
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
