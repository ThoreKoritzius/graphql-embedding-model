"""Find demo examples with high wow-factor:
  - target owner is an everyday concept (Order, Room, Customer, ...)
  - base model puts target at rank >= 20 (clear mess-up)
  - base model's top-1 IS a same-field-name sibling (owner disambig story)
  - tuned model puts target at rank 1
  - both target and distractor owners are human-readable

Prints the top candidates. Pick one and paste the full query + target into
demo_visualize.py / demo_ladder.py to generate plots.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


FRIENDLY_OWNERS = {
    "order", "cart", "product", "customer", "user", "post", "comment",
    "issue", "article", "review", "booking", "reservation", "hotel",
    "room", "guest", "payment", "invoice", "vehicle", "listing", "dealer",
    "flight", "passenger", "ticket", "trip", "message", "conversation",
    "account", "transaction", "subscription", "shipment", "address",
    "contact", "employee", "task", "event", "notification", "inventory",
    "appointment", "prescription", "patient", "doctor", "provider",
    "merchant", "refund", "discount", "coupon", "return", "brand",
    "collection", "wishlist",
}

# Ugly synthetic artifacts to filter out.
UGLY_PATTERNS = [
    re.compile(r"Extension\d+", re.IGNORECASE),
    re.compile(r"^[a-z]+Input$", re.IGNORECASE),
    re.compile(r"Payload$"),
    re.compile(r"Connection$"),
    re.compile(r"Edge$"),
]


def owner_is_friendly(owner: str) -> bool:
    if any(p.search(owner) for p in UGLY_PATTERNS):
        return False
    tokens = re.findall(r"[A-Z][a-z]+|[a-z]+", owner)
    return any(t.lower() in FRIENDLY_OWNERS for t in tokens)


def load_corpus(path: Path):
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
    ap.add_argument("--min-base-rank", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    coords, texts = load_corpus(Path(args.corpus))
    coord_idx = {c: i for i, c in enumerate(coords)}
    print(f"encoding {len(coords)} docs with both models...")
    base_corpus = encode_with_resolution(args.base_model, texts, prompt_name="document")
    tuned_corpus = encode_with_resolution(args.tuned_model, texts, prompt_name="document")

    rows = [json.loads(l) for l in Path(args.queries).open()]
    print(f"encoding {len(rows)} queries...")
    queries = [r["query"] for r in rows]
    base_q = encode_with_resolution(args.base_model, queries, prompt_name="query")
    tuned_q = encode_with_resolution(args.tuned_model, queries, prompt_name="query")

    cands = []
    for i, r in enumerate(rows):
        target = r["positive_coordinate"]
        if target not in coord_idx:
            continue
        tgt = coord_idx[target]
        owner, field = target.split(".", 1)
        if not owner_is_friendly(owner):
            continue
        # Skip queries that literally name the field.
        if field.lower() in r["query"].lower():
            continue

        bsim = base_corpus @ base_q[i]
        tsim = tuned_corpus @ tuned_q[i]
        brank = int(np.sum(bsim > bsim[tgt])) + 1
        trank = int(np.sum(tsim > tsim[tgt])) + 1
        if trank != 1 or brank < args.min_base_rank:
            continue

        top1_coord = coords[int(np.argmax(bsim))]
        top1_owner, top1_field = top1_coord.split(".", 1)
        # Same-field-name collision is the ideal owner-disambiguation story.
        # Relax: also allow different field on a friendly owner (the model
        # picked the wrong *concept entirely*), still makes the lift point.
        same_field_sibling = (top1_field.lower() == field.lower())
        friendly_top1 = owner_is_friendly(top1_owner)
        if not (same_field_sibling or friendly_top1):
            continue

        cands.append({
            "query": r["query"],
            "target": target,
            "base_top1": top1_coord,
            "base_rank": brank,
            "tuned_rank": trank,
            "query_len": len(r["query"]),
            "same_field_sibling": same_field_sibling,
            "friendly_top1": friendly_top1,
        })

    # Prefer same-field-sibling stories (strongest narrative), then big rank flip.
    cands.sort(key=lambda c: (not c["same_field_sibling"], -c["base_rank"], c["query_len"]))
    print(f"\n{len(cands)} candidates found. Top {args.top_k}:\n")
    for c in cands[: args.top_k]:
        flag = "[sibling]" if c["same_field_sibling"] else "[friendly-owner]"
        print(f"  {flag}  Q: {c['query']}")
        print(f"     target:    {c['target']}")
        print(f"     base pick: {c['base_top1']}  (rank {c['base_rank']} → {c['tuned_rank']})")
        print()


if __name__ == "__main__":
    main()
