from __future__ import annotations

import math
from collections.abc import Iterable


def recall_at_k(ranked: list[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def mrr_at_k(ranked: list[str], target: str, k: int) -> float:
    for idx, item in enumerate(ranked[:k], start=1):
        if item == target:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked: list[str], target: str, k: int) -> float:
    for idx, item in enumerate(ranked[:k], start=1):
        if item == target:
            dcg = 1.0 / math.log2(idx + 1)
            idcg = 1.0
            return dcg / idcg
    return 0.0


def aggregate(metric_values: Iterable[float]) -> float:
    vals = list(metric_values)
    return sum(vals) / len(vals) if vals else 0.0
