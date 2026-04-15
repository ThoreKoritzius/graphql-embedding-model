from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord


def _light_embed(texts: list[str]) -> np.ndarray:
    dims = 512
    arr = np.zeros((len(texts), dims), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in t.lower().split():
            arr[i, hash(tok) % dims] += 1.0
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def mine_hard_negatives(corpus: list[CorpusRecord], queries: list[QueryRecord], hard_k: int = 5, easy_k: int = 5, medium_k: int = 5) -> list[QueryRecord]:
    corpus_ids = [c.coordinate for c in corpus]
    corpus_texts = [c.full_text for c in corpus]
    corpus_emb = _light_embed(corpus_texts)
    by_coord = {c.coordinate: c for c in corpus}

    out: list[QueryRecord] = []
    for q in queries:
        q_emb = _light_embed([q.query])
        sims = cosine_similarity(q_emb, corpus_emb)[0]
        ranked_idx = np.argsort(-sims)

        hard: list[str] = []
        medium: list[str] = []
        easy: list[str] = []
        target = by_coord.get(q.positive_coordinate)
        for idx in ranked_idx:
            cid = corpus_ids[idx]
            if cid == q.positive_coordinate:
                continue
            cand = by_coord[cid]
            if target and cand.owner_type == target.owner_type and len(hard) < hard_k:
                hard.append(cid)
            elif target and cand.field_name == target.field_name and len(medium) < medium_k:
                medium.append(cid)
            elif len(easy) < easy_k:
                easy.append(cid)
            if len(hard) >= hard_k and len(medium) >= medium_k and len(easy) >= easy_k:
                break

        out.append(q.model_copy(update={"negatives_easy": easy, "negatives_medium": medium, "negatives_hard": hard, "negative_coordinates": list(dict.fromkeys(hard + medium + easy))[: hard_k + medium_k + easy_k]}))
    return out
