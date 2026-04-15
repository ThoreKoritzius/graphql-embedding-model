from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphql_finetuning_pipeline.data.models import CorpusRecord
from graphql_finetuning_pipeline.data.structural_views import ensure_view_available, get_view_text, normalize_primary_retrieval_view
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution
from graphql_finetuning_pipeline.utils.io import ensure_dir


def build_index(
    corpus_rows: list[CorpusRecord],
    model_path_or_name: str,
    out_dir: Path,
    retrieval_view: str = "sdl",
) -> dict:
    ensure_dir(out_dir)
    view = normalize_primary_retrieval_view(retrieval_view)
    ensure_view_available(corpus_rows, view)

    ids = [c.coordinate for c in corpus_rows]
    texts = [get_view_text(c, view) for c in corpus_rows]
    emb = encode_with_resolution(model_path_or_name, texts, allow_remote_fallback=True)

    config = {
        "top_k_default": 10,
        "normalize_embeddings": True,
        "similarity": "cosine",
        "index_type": "sklearn-bruteforce",
        "truncation_length": 512,
        "retrieval_view": view,
    }

    # Attempt FAISS first.
    try:
        import faiss

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        config["index_type"] = "faiss-flatip"
    except Exception:
        with (out_dir / "index.pkl").open("wb") as f:
            pickle.dump({"ids": ids, "embeddings": emb}, f)

    (out_dir / "ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")
    (out_dir / "retrieval_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def search_local(index_dir: Path, query_emb: np.ndarray, top_k: int = 10) -> list[list[str]]:
    ids = json.loads((index_dir / "ids.json").read_text(encoding="utf-8"))

    faiss_path = index_dir / "index.faiss"
    if faiss_path.exists():
        import faiss

        index = faiss.read_index(str(faiss_path))
        _, idx = index.search(query_emb.astype(np.float32), top_k)
        return [[ids[j] for j in row] for row in idx]

    with (index_dir / "index.pkl").open("rb") as f:
        payload = pickle.load(f)
    emb = payload["embeddings"]
    sims = cosine_similarity(query_emb, emb)
    rows = np.argsort(-sims, axis=1)[:, :top_k]
    return [[ids[j] for j in row] for row in rows]
