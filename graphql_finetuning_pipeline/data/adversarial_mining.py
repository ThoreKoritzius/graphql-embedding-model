"""Hard-example mining against the cold base embedding model.

Every validated candidate query is encoded by the base checkpoint we're
about to fine-tune from (default Qwen3-Embedding-0.6B) and scored against
its competition set. Rows where the base model ranks the wrong coordinate
first are the highest-value training examples — they're exactly the
failure modes fine-tuning must correct.

The contract with ``openai_seed.generate_openai_seed`` is deliberately
thin: score a batch, tag each item with bucket + base-top-1-wrong for
downstream negative-mining, then the caller stratifies. Keeping the
stratification in one place (``stratify``) lets downstream metrics
audit the bucket mix without reimplementing it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from graphql_finetuning_pipeline.data.openai_seed import SeedItem
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


@dataclass
class ScoredItem:
    item: SeedItem
    bucket: str  # "hard" | "medium" | "easy"
    base_top1_correct: bool
    base_target_rank: int
    base_margin: float
    base_top1_wrong_coord: str | None  # the coordinate the base model incorrectly ranked first


def _render_competitor_text(field: dict) -> str:
    """Turn a field_catalog entry into a document-side encoding string.

    Using owner description + field signature mirrors how the retrieval
    corpus is built (see ``data/corpus.py:field_semantic_text``) so the
    mining signal tracks what the actual retriever sees.
    """
    owner = (field.get("owner_description") or "").strip()
    field_desc = (field.get("description") or "").strip()
    rt = field.get("return_type") or ""
    coord = field["coordinate"]
    parts = [f"Field {coord}", f"Returns: {rt}"]
    if owner:
        parts.append(f"Owner: {owner}")
    if field_desc:
        parts.append(f"Description: {field_desc}")
    return ". ".join(parts)


def score_candidates(
    items: list[SeedItem],
    competition_set: list[dict],
    *,
    base_model: str,
    margin_threshold: float = 0.05,
) -> list[ScoredItem]:
    """Encode queries + competitor docs with the base model; bucket each row.

    ``competition_set`` is the list of field_catalog entries the prompt was
    built around (one per competing owner type). Each item's positive
    coordinate must appear in this set — otherwise we can't score it. Rows
    whose positive is missing are returned with bucket="easy" and logged
    implicitly via ``base_target_rank == -1``; they'll still feed the
    easy floor so the pipeline doesn't lose rows.
    """
    if not items:
        return []
    if len(competition_set) < 2:
        # Can't mine against fewer than 2 candidates; everything is easy by default.
        return [
            ScoredItem(item=it, bucket="easy", base_top1_correct=True,
                       base_target_rank=0, base_margin=1.0, base_top1_wrong_coord=None)
            for it in items
        ]
    coord_to_idx = {f["coordinate"]: i for i, f in enumerate(competition_set)}
    q_emb = encode_with_resolution(base_model, [it.query for it in items], prompt_name="query")
    c_emb = encode_with_resolution(base_model, [_render_competitor_text(f) for f in competition_set], prompt_name="document")
    # Normalized already via encode_with_resolution's normalize_embeddings=True.
    sims = q_emb @ c_emb.T  # (n_items, n_competitors)

    out: list[ScoredItem] = []
    for i, item in enumerate(items):
        target_idx = coord_to_idx.get(item.positive_coordinate, -1)
        if target_idx < 0:
            out.append(ScoredItem(item=item, bucket="easy", base_top1_correct=True,
                                  base_target_rank=-1, base_margin=1.0,
                                  base_top1_wrong_coord=None))
            continue
        row = sims[i]
        ranked = np.argsort(row)[::-1]
        top1_idx = int(ranked[0])
        top1_correct = top1_idx == target_idx
        target_rank = int(np.where(ranked == target_idx)[0][0])
        if top1_correct:
            runner_up_idx = int(ranked[1]) if len(ranked) > 1 else target_idx
            margin = float(row[target_idx] - row[runner_up_idx])
        else:
            margin = float(row[target_idx] - row[top1_idx])
        if not top1_correct:
            bucket = "hard"
            wrong_coord = competition_set[top1_idx]["coordinate"]
        elif margin < margin_threshold:
            bucket = "medium"
            wrong_coord = None
        else:
            bucket = "easy"
            wrong_coord = None
        out.append(ScoredItem(
            item=item, bucket=bucket, base_top1_correct=top1_correct,
            base_target_rank=target_rank, base_margin=margin,
            base_top1_wrong_coord=wrong_coord,
        ))
    return out


def stratify(
    scored: list[ScoredItem],
    *,
    target_hard_fraction: float,
    target_medium_fraction: float,
) -> list[ScoredItem]:
    """Downsample to the requested bucket mix without ever upsampling.

    Keeping an invariant of "never invent rows" — we always return a subset
    of the input. If one bucket is empty we let the others fill the gap
    rather than crashing; the verification step logs the actual mix so the
    operator can spot the imbalance.
    """
    if not scored:
        return []
    easy_fraction = max(0.0, 1.0 - target_hard_fraction - target_medium_fraction)
    by_bucket: dict[str, list[ScoredItem]] = {"hard": [], "medium": [], "easy": []}
    for s in scored:
        by_bucket[s.bucket].append(s)

    # The scarcest bucket — weighted by target share — sets the total.
    capacities = []
    for bucket, frac in [("hard", target_hard_fraction), ("medium", target_medium_fraction), ("easy", easy_fraction)]:
        avail = len(by_bucket[bucket])
        if frac <= 0 or avail == 0:
            continue
        capacities.append(avail / frac)
    if not capacities:
        return scored  # degenerate: nothing fits the stratification; return all
    total = int(min(capacities))
    if total <= 0:
        return scored

    out: list[ScoredItem] = []
    for bucket, frac in [("hard", target_hard_fraction), ("medium", target_medium_fraction), ("easy", easy_fraction)]:
        take = int(round(total * frac))
        if take <= 0:
            continue
        pool = by_bucket[bucket]
        out.extend(pool[:take])
    return out


def bucket_counts(scored: list[ScoredItem]) -> dict[str, int]:
    counts = {"hard": 0, "medium": 0, "easy": 0}
    for s in scored:
        counts[s.bucket] = counts.get(s.bucket, 0) + 1
    return counts
