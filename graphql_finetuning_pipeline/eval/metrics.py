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


def set_recall_at_k(ranked: list[str], relevant: list[str], k: int, mode: str = "any") -> float:
    rel = [x for x in relevant if x]
    if not rel:
        return 0.0
    top = set(ranked[:k])
    if mode == "all":
        return 1.0 if all(x in top for x in rel) else 0.0
    return 1.0 if any(x in top for x in rel) else 0.0


def coverage_at_k(ranked: list[str], relevant: list[str], k: int) -> float:
    rel = [x for x in relevant if x]
    if not rel:
        return 0.0
    top = set(ranked[:k])
    hit = sum(1 for x in rel if x in top)
    return hit / len(rel)


def pair_recall_at_k(ranked: list[str], primary: str, bridge: str, k: int) -> float:
    top = set(ranked[:k])
    if not primary or not bridge:
        return 0.0
    return 1.0 if primary in top and bridge in top else 0.0
