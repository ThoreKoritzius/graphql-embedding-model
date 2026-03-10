from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord


def _light_embed(texts: list[str]) -> np.ndarray:
    # Cheap deterministic fallback embedding.
    dims = 512
    arr = np.zeros((len(texts), dims), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            arr[i, hash(tok) % dims] += 1.0
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def mine_hard_negatives(
    corpus: list[CorpusRecord],
    queries: list[QueryRecord],
    hard_k: int = 5,
    easy_k: int = 5,
    medium_k: int = 5,
) -> list[QueryRecord]:
    corpus_ids = [c.type_id for c in corpus]
    corpus_texts = [c.full_text for c in corpus]
    corpus_emb = _light_embed(corpus_texts)

    out: list[QueryRecord] = []
    for q in queries:
        q_emb = _light_embed([q.query])
        sims = cosine_similarity(q_emb, corpus_emb)[0]
        ranked_idx = np.argsort(-sims)

        hard: list[str] = []
        for idx in ranked_idx:
            cid = corpus_ids[idx]
            if cid == q.target_type_id:
                continue
            hard.append(cid)
            if len(hard) >= hard_k:
                break

        easy_pool = [cid for cid in corpus_ids if cid != q.target_type_id and cid not in hard]
        easy = easy_pool[:easy_k]

        medium = []
        target_prefix = q.target_type_id[:3].lower()
        for cid in corpus_ids:
            if cid == q.target_type_id or cid in hard:
                continue
            if cid[:3].lower() == target_prefix:
                medium.append(cid)
            if len(medium) >= medium_k:
                break

        out.append(
            q.model_copy(
                update={
                    "negatives_easy": easy,
                    "negatives_medium": medium,
                    "negatives_hard": hard,
                }
            )
        )
    return out
