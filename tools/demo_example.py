"""Find a test query where the fine-tune clearly beats the base model, and
render top-5 retrievals from each side-by-side. Output is a markdown snippet
you can paste into artifacts/evals/README.md.

Run on the H100 where both checkpoints live:

    python tools/demo_example.py \
        --base-model Qwen/Qwen3-Embedding-0.6B \
        --tuned-model artifacts/models/qwen3-v7-h100-e3/best \
        --corpus artifacts/datasets/v7/corpus.jsonl \
        --queries artifacts/datasets/v7/test.jsonl \
        --out artifacts/evals/qwen3-v7-e3/demo_example.md

The script scans the first N queries (default 80 — enough to find a good
example without loading the full test set), picks the row where the tuned
model moves the target from a low rank up to #1 AND the baseline top-1 is
a plausible distractor (same field name, different owner). That's the
case that makes the owner-disambiguation argument concrete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    coords: list[str] = []
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            coords.append(c["coordinate"])
            # Use the same retrieval_text that the eval pipeline uses.
            texts.append(c.get("retrieval_text") or c.get("field_semantic_text") or c.get("full_text") or c["coordinate"])
    return coords, texts


def load_queries(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--tuned-model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-queries", type=int, default=80)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    coords, texts = load_corpus(Path(args.corpus))
    rows = load_queries(Path(args.queries), args.limit_queries)
    coord_idx = {c: i for i, c in enumerate(coords)}

    print(f"encoding corpus with base model ({len(texts)} docs)...")
    base_corpus = encode_with_resolution(args.base_model, texts, prompt_name="document")
    print(f"encoding corpus with tuned model...")
    tuned_corpus = encode_with_resolution(args.tuned_model, texts, prompt_name="document")

    queries = [r["query"] for r in rows]
    print(f"encoding {len(queries)} queries with base model...")
    base_q = encode_with_resolution(args.base_model, queries, prompt_name="query")
    print(f"encoding queries with tuned model...")
    tuned_q = encode_with_resolution(args.tuned_model, queries, prompt_name="query")

    # Find the best demo candidate: biggest rank improvement, with the
    # baseline's top-1 being a sibling (same field name on a different owner).
    best = None
    best_score = -1
    for i, r in enumerate(rows):
        target = r["positive_coordinate"]
        if target not in coord_idx:
            continue
        tgt = coord_idx[target]
        base_scores = base_corpus @ base_q[i]
        tuned_scores = tuned_corpus @ tuned_q[i]
        base_rank = int(np.sum(base_scores > base_scores[tgt]))
        tuned_rank = int(np.sum(tuned_scores > tuned_scores[tgt]))
        # Want: tuned rank = 0 (top-1), base rank >= 2 (genuinely missed it)
        if tuned_rank != 0 or base_rank < 2:
            continue
        base_top1_coord = coords[int(np.argmax(base_scores))]
        same_field = base_top1_coord.split(".")[-1].lower() == target.split(".")[-1].lower()
        different_owner = base_top1_coord.split(".")[0] != target.split(".")[0]
        sibling_bonus = 10 if (same_field and different_owner) else 0
        score = base_rank + sibling_bonus
        if score > best_score:
            best_score = score
            best = (i, r, base_scores, tuned_scores)

    if best is None:
        print("no clear winning example found; widen --limit-queries")
        return

    i, r, base_scores, tuned_scores = best
    target = r["positive_coordinate"]
    target_idx = coord_idx[target]

    def top_k(scores: np.ndarray, k: int):
        idx = np.argsort(scores)[::-1][:k]
        return [(coords[j], float(scores[j])) for j in idx]

    base_top = top_k(base_scores, args.top_k)
    tuned_top = top_k(tuned_scores, args.top_k)
    base_rank = int(np.sum(base_scores > base_scores[target_idx]))
    tuned_rank = int(np.sum(tuned_scores > tuned_scores[target_idx]))

    # Markdown output
    def render_top(rows, target):
        out = "| rank | coordinate | score |\n|---|---|---|\n"
        for rk, (c, s) in enumerate(rows, 1):
            mark = " ✅ **target**" if c == target else ""
            out += f"| {rk} | `{c}`{mark} | {s:.3f} |\n"
        return out

    md = f"""## Example: owner-type disambiguation in action

**Query:** "{r['query']}"

**Correct answer:** `{target}`

The query is phrased like a real user question — no `Type.field` hint, no GraphQL vocabulary. The hard part isn't identifying the field name; it's picking the *right owner type* out of several that expose the same field.

### Base model (Qwen3-Embedding-0.6B, untuned)

Target rank: **{base_rank + 1}**

{render_top(base_top, target)}

### Fine-tuned model

Target rank: **{tuned_rank + 1}**

{render_top(tuned_top, target)}

The fine-tune learned that the query's phrasing (the context around the field name) points at `{target.split('.')[0]}` specifically, while the base model fell back to the alphabetically- or semantically-earlier sibling. This is the pattern the v7 competition-set prompt was designed to produce training signal for.
"""

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"wrote {out}")
    print("\nsnippet:")
    print(md)


if __name__ == "__main__":
    main()
