from __future__ import annotations

from pathlib import Path

import numpy as np


def light_embed(texts: list[str], dims: int = 512) -> np.ndarray:
    arr = np.zeros((len(texts), dims), dtype=np.float32)
    for i, text in enumerate(texts):
        for tok in text.lower().split():
            arr[i, hash(tok) % dims] += 1.0
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms


def is_local_model_reference(model_ref: str) -> bool:
    p = Path(model_ref).expanduser()
    if p.exists():
        return True
    if model_ref.startswith((".", "/", "~")):
        return True
    # HF IDs are typically "org/model" (single slash). Deeper slash depth is usually local.
    return model_ref.count("/") >= 2


def _validate_local_model_dir(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"Local model path exists but is not a directory: {path}")
    required = ["modules.json", "config_sentence_transformers.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Local model directory is missing required SentenceTransformer files ({missing_str}): {path}. "
            "Use the output directory from a successful `graphft train-embedder` run."
        )


def encode_with_resolution(
    model_ref: str,
    texts: list[str],
    *,
    allow_remote_fallback: bool = True,
) -> np.ndarray:
    model_path = Path(model_ref).expanduser()
    local_ref = is_local_model_reference(model_ref)

    if local_ref and not model_path.exists():
        raise FileNotFoundError(
            f"Local model path does not exist: {model_path}. "
            "Run `graphft train-embedder` and pass its `--out-dir` to this command."
        )
    if local_ref and model_path.exists():
        _validate_local_model_dir(model_path)

    try:
        from sentence_transformers import SentenceTransformer

        resolved = str(model_path) if local_ref else model_ref
        model = SentenceTransformer(resolved)
        emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
    except Exception as exc:
        if local_ref:
            raise RuntimeError(f"Failed to load local SentenceTransformer model from {model_path}: {exc}") from exc
        if allow_remote_fallback:
            return light_embed(texts)
        raise
