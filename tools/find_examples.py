"""Find the top-K test queries where the fine-tune most dramatically improves
retrieval AND the owner/field vocabulary is easy for a lay reader to grok.

For each query: compute (base_rank, tuned_rank), score it on:
  - tuned must be top-1 (clean narrative)
  - rank improvement (higher = more dramatic story)
  - owner names must be common English (no SlaPolicy, no CartExtension3)
  - query must not already contain the field name literal

Prints the top candidates so you can eyeball and pick one.

Run on the H100:

    python tools/find_examples.py \
      --base-model Qwen/Qwen3-Embedding-0.6B \
      --tuned-model artifacts/models/qwen3-v7-h100-e3/best \
      --corpus artifacts/datasets/v7/corpus.jsonl \
      --queries artifacts/datasets/v7/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


# "Easy for a lay reader" — common business-domain nouns.
FRIENDLY_OWNERS = {
    "order", "cart", "product", "customer", "user", "post", "comment",
    "issue", "article", "review", "booking", "reservation", "hotel",
    "room", "guest", "payment", "invoice", "vehicle", "listing", "dealer",
    "flight", "passenger", "ticket", "trip", "message", "conversation",
    "account", "transaction", "subscription", "shipment", "address",
    "contact", "employee", "task", "event", "notification", "inventory",
}


def owner_is_friendly(owner: str) -> bool:
    # Split camelCase and check if any token is in the friendly list.
    tokens = re.findall(r"[A-Z][a-z]+|[a-z]+", owner)
    return any(t.lower() in FRIENDLY_OWNERS for t in tokens)


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    coords, texts, seen = [], [], set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            coord = c["coordinate"]
            if coord in seen:
                continue
            seen.add(coord)
            coords.append(coord)
            texts.append(c.get("retrieval_text") or c.get("field_semantic_text") or c.get("full_text") or coord)
    return coords, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--tuned-model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--top-k", type=int, default=10, help="How many top candidates to print.")
    ap.add_argument("--limit-queries", type=int, default=223, help="Scan window.")
    args = ap.parse_args()

    coords, texts = load_corpus(Path(args.corpus))
    coord_idx = {c: i for i, c in enumerate(coords)}
    print(f"encoding {len(coords)} corpus docs with both models...")
    base_corpus = encode_with_resolution(args.base_model, texts, prompt_name="document")
    tuned_corpus = encode_with_resolution(args.tuned_model, texts, prompt_name="document")

    rows = []
    with Path(args.queries).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit_queries:
                break
            rows.append(json.loads(line))
    print(f"encoding {len(rows)} queries...")
    queries = [r["query"] for r in rows]
    base_q = encode_with_resolution(args.base_model, queries, prompt_name="query")
    tuned_q = encode_with_resolution(args.tuned_model, queries, prompt_name="query")

    # Score every candidate.
    candidates = []
    for i, r in enumerate(rows):
        target = r["positive_coordinate"]
        if target not in coord_idx:
            continue
        tgt = coord_idx[target]
        base_scores = base_corpus @ base_q[i]
        tuned_scores = tuned_corpus @ tuned_q[i]
        base_rank = int(np.sum(base_scores > base_scores[tgt])) + 1
        tuned_rank = int(np.sum(tuned_scores > tuned_scores[tgt])) + 1
        if tuned_rank != 1:
            continue

        owner, field = target.split(".", 1)
        if not owner_is_friendly(owner):
            continue
        # Skip queries that contain the field name literally — not interesting.
        if field.lower() in r["query"].lower():
            continue
        # Sibling count with same field name.
        siblings = [c for c in coords if c != target and c.split(".", 1)[1].lower() == field.lower()]
        if len(siblings) < 3:
            continue

        base_top1_coord = coords[int(np.argmax(base_scores))]
        base_top1_owner = base_top1_coord.split(".")[0]
        base_top1_friendly = owner_is_friendly(base_top1_owner)

        candidates.append({
            "query": r["query"],
            "target": target,
            "target_owner_friendly": True,
            "base_top1": base_top1_coord,
            "base_top1_friendly": base_top1_friendly,
            "base_rank": base_rank,
            "tuned_rank": tuned_rank,
            "rank_improvement": base_rank - tuned_rank,
            "n_siblings": len(siblings),
            "base_top1_is_sibling": base_top1_coord in siblings,
        })

    # Rank candidates: prefer (target and base-top1 both friendly) + big rank improvement
    # + base-top1 is actually a same-field sibling (strongest narrative).
    def score(c):
        s = c["rank_improvement"]
        if c["base_top1_is_sibling"]:
            s += 50
        if c["base_top1_friendly"]:
            s += 30
        return s

    candidates.sort(key=score, reverse=True)

    print(f"\nTop {args.top_k} candidate queries (friendly owner + big rank flip):\n")
    for c in candidates[: args.top_k]:
        print(f"  Q: {c['query']}")
        print(f"    target:     {c['target']}")
        print(f"    base top1:  {c['base_top1']}  (sibling={c['base_top1_is_sibling']})")
        print(f"    base rank:  {c['base_rank']}  →  tuned rank: {c['tuned_rank']}")
        print(f"    siblings:   {c['n_siblings']}")
        print()


if __name__ == "__main__":
    main()
