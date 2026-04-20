"""Ranking-ladder plot: shows where the target lands among its siblings
under base vs tuned model, using real cosine similarities (no projection
distortion).

For one query, take every sibling coordinate (same field name, different
owner types). Plot their cosines to the query as a vertical ladder per
model. Target line is highlighted. Reader sees the target literally climb
from "buried among distractors" to "alone at the top".

Usage on the H100 (same args as demo_visualize.py):

    python tools/demo_ladder.py \
      --base-model Qwen/Qwen3-Embedding-0.6B \
      --tuned-model artifacts/models/qwen3-v7-h100-e3/best \
      --corpus artifacts/datasets/v7/corpus.jsonl \
      --query "I need to understand what commitments we have regarding support response times. Where can I find that info?" \
      --target SlaPolicy.description \
      --out artifacts/evals/plots/ranking_ladder.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


def _wrap(text: str, width: int = 90) -> str:
    """Minimal word-wrap for matplotlib titles."""
    words = text.split()
    out, line = [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    coords: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
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
    ap.add_argument("--query", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default="artifacts/evals/plots/ranking_ladder.png")
    # Show this many siblings (ranked by base-model cosine — i.e. the
    # most-confusing distractors by baseline standards).
    ap.add_argument("--max-siblings", type=int, default=15)
    args = ap.parse_args()

    print("loading corpus + encoding...")
    coords, texts = load_corpus(Path(args.corpus))
    if args.target not in coords:
        raise SystemExit(f"target {args.target} not in deduped corpus")
    target_field = args.target.split(".", 1)[1].lower()
    siblings = [c for c in coords if c != args.target and c.split(".", 1)[1].lower() == target_field]
    all_candidates = [args.target] + siblings
    cand_texts = [texts[coords.index(c)] for c in all_candidates]

    # Encode only the candidate set + query — no need for the full corpus.
    base_docs = encode_with_resolution(args.base_model, cand_texts, prompt_name="document")
    tuned_docs = encode_with_resolution(args.tuned_model, cand_texts, prompt_name="document")
    base_q = encode_with_resolution(args.base_model, [args.query], prompt_name="query")[0]
    tuned_q = encode_with_resolution(args.tuned_model, [args.query], prompt_name="query")[0]

    base_sims = base_docs @ base_q
    tuned_sims = tuned_docs @ tuned_q

    # Keep the top-K by *baseline* similarity — these are the actual
    # confusers. Always keep the target regardless of its baseline rank.
    top_base = np.argsort(base_sims)[::-1][: args.max_siblings]
    keep = sorted(set(top_base.tolist() + [0]))  # index 0 = target
    labels = [all_candidates[i] for i in keep]
    bvals = base_sims[keep]
    tvals = tuned_sims[keep]

    def render_panel(ax, sims, labels, title, target_label):
        order = np.argsort(sims)[::-1]
        for rank, idx in enumerate(order):
            y = len(order) - rank - 1  # top = best
            label = labels[idx]
            is_target = (label == target_label)
            color = "#2ecc71" if is_target else "#888"
            weight = "bold" if is_target else "normal"
            ax.hlines(y, 0, sims[idx], colors=color, linewidth=6 if is_target else 3, alpha=0.95 if is_target else 0.55)
            ax.plot(sims[idx], y, "o", color=color, markersize=9 if is_target else 6)
            owner = label.split(".")[0]
            ax.text(sims[idx] + 0.005, y, f"{owner}  ({sims[idx]:.2f})",
                    va="center", fontsize=9, color=("#196f3d" if is_target else "#444"),
                    fontweight=weight)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"cosine(query, <owner>.{args.target.split('.')[1]})")
        ax.set_yticks([])
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.set_xlim(0, max(sims.max() * 1.35, 0.55))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.25)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, 0.42 * len(keep) + 2.5)), sharey=True)
    render_panel(axes[0], bvals, labels, "Base model (Qwen3-0.6B)", args.target)
    render_panel(axes[1], tvals, labels, "Fine-tuned model", args.target)

    wrapped = _wrap(args.query, width=90)
    fig.suptitle(
        f'"{wrapped}"',
        fontsize=12, y=1.02, ha="center", fontstyle="italic",
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")

    # Print the rank flip so we can cite real numbers in the README.
    base_rank = int(np.sum(base_sims > base_sims[0])) + 1
    tuned_rank = int(np.sum(tuned_sims > tuned_sims[0])) + 1
    print(json.dumps({
        "candidates_considered": len(all_candidates),
        "target_base_rank_among_siblings": base_rank,
        "target_tuned_rank_among_siblings": tuned_rank,
        "target_base_cosine": round(float(base_sims[0]), 4),
        "target_tuned_cosine": round(float(tuned_sims[0]), 4),
    }, indent=2))


if __name__ == "__main__":
    main()
