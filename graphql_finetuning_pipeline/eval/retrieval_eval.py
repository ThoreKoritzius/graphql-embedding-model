from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.eval.metrics import (
    aggregate,
    coverage_at_k,
    mrr_at_k,
    ndcg_at_k,
    pair_recall_at_k,
    recall_at_k,
    set_recall_at_k,
)
from graphql_finetuning_pipeline.data.structural_views import ensure_view_available, get_view_text, normalize_primary_retrieval_view
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution
from graphql_finetuning_pipeline.utils.io import ensure_dir


def evaluate(
    eval_rows: list[QueryRecord],
    corpus_rows: list[CorpusRecord],
    model_path_or_name: str,
    out_path: Path | None = None,
    retrieval_view: str = "sdl",
) -> dict:
    if not eval_rows:
        metrics = {
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
        }
        if out_path:
            ensure_dir(out_path.parent)
            out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    view = normalize_primary_retrieval_view(retrieval_view)
    ensure_view_available(corpus_rows, view)
    corpus_ids = [c.type_id for c in corpus_rows]
    corpus_texts = [get_view_text(c, view) for c in corpus_rows]

    t0 = time.perf_counter()
    corpus_emb = encode_with_resolution(model_path_or_name, corpus_texts, allow_remote_fallback=True)
    t1 = time.perf_counter()
    query_emb = encode_with_resolution(
        model_path_or_name,
        [q.query for q in eval_rows],
        allow_remote_fallback=True,
    )
    t2 = time.perf_counter()

    sims = cosine_similarity(query_emb, corpus_emb)
    r1, r5, r10, mrr10, ndcg10 = [], [], [], [], []
    set_any5, set_all10, cov10, pair10 = [], [], [], []

    for i, row in enumerate(eval_rows):
        ranked_idx = np.argsort(-sims[i])
        ranked_ids = [corpus_ids[j] for j in ranked_idx]
        r1.append(recall_at_k(ranked_ids, row.target_type_id, 1))
        r5.append(recall_at_k(ranked_ids, row.target_type_id, 5))
        r10.append(recall_at_k(ranked_ids, row.target_type_id, 10))
        mrr10.append(mrr_at_k(ranked_ids, row.target_type_id, 10))
        ndcg10.append(ndcg_at_k(ranked_ids, row.target_type_id, 10))
        relevant = row.relevant_type_ids or [row.primary_type_id or row.target_type_id]
        set_any5.append(set_recall_at_k(ranked_ids, relevant, 5, mode="any"))
        set_all10.append(set_recall_at_k(ranked_ids, relevant, 10, mode="all"))
        cov10.append(coverage_at_k(ranked_ids, relevant, 10))
        pair = row.relation_pair or {}
        pair10.append(pair_recall_at_k(ranked_ids, pair.get("primary", ""), pair.get("bridge", ""), 10))

    metrics = {
        "count": len(eval_rows),
        "recall@1": aggregate(r1),
        "recall@5": aggregate(r5),
        "recall@10": aggregate(r10),
        "mrr@10": aggregate(mrr10),
        "ndcg@10": aggregate(ndcg10),
        "set_recall_any@5": aggregate(set_any5),
        "set_recall_all@10": aggregate(set_all10),
        "coverage@10": aggregate(cov10),
        "pair_recall@10": aggregate(pair10),
        "embedding_latency_seconds": {
            "corpus": t1 - t0,
            "queries": t2 - t1,
        },
    }

    if out_path:
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
