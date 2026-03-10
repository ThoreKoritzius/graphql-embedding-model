from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.eval.metrics import aggregate, mrr_at_k, ndcg_at_k, recall_at_k
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution
from graphql_finetuning_pipeline.utils.io import ensure_dir


def evaluate(
    eval_rows: list[QueryRecord],
    corpus_rows: list[CorpusRecord],
    model_path_or_name: str,
    out_path: Path | None = None,
) -> dict:
    corpus_ids = [c.type_id for c in corpus_rows]
    corpus_texts = [c.full_text for c in corpus_rows]

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

    for i, row in enumerate(eval_rows):
        ranked_idx = np.argsort(-sims[i])
        ranked_ids = [corpus_ids[j] for j in ranked_idx]
        r1.append(recall_at_k(ranked_ids, row.target_type_id, 1))
        r5.append(recall_at_k(ranked_ids, row.target_type_id, 5))
        r10.append(recall_at_k(ranked_ids, row.target_type_id, 10))
        mrr10.append(mrr_at_k(ranked_ids, row.target_type_id, 10))
        ndcg10.append(ndcg_at_k(ranked_ids, row.target_type_id, 10))

    metrics = {
        "count": len(eval_rows),
        "recall@1": aggregate(r1),
        "recall@5": aggregate(r5),
        "recall@10": aggregate(r10),
        "mrr@10": aggregate(mrr10),
        "ndcg@10": aggregate(ndcg10),
        "embedding_latency_seconds": {
            "corpus": t1 - t0,
            "queries": t2 - t1,
        },
    }

    if out_path:
        ensure_dir(out_path.parent)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
