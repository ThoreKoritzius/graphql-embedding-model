"""Visualize one test query's neighborhood in embedding space.

Two complementary plots:
  1. embedding_space.png — PCA panels (base vs tuned) showing query, target,
     sibling distractors (same field name), and random non-siblings. Reader
     sees the sibling cluster reorganize after fine-tuning.
  2. sibling_cosines.png — bar chart of cosine(query, coord) for target +
     siblings. Baseline grey, tuned blue. Makes the ranking flip concrete.

Run on the H100 after training:

    python tools/demo_visualize.py \
        --base-model Qwen/Qwen3-Embedding-0.6B \
        --tuned-model artifacts/models/qwen3-v7-h100-e3/best \
        --corpus artifacts/datasets/v7/corpus.jsonl \
        --query "I need to understand what commitments we have regarding support response times. Where can I find that info?" \
        --target SlaPolicy.description \
        --out-dir artifacts/evals/plots

If --query/--target aren't provided, the script falls back to picking the
demo example itself (same logic as tools/demo_example.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    coords: list[str] = []
    texts: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            coord = c["coordinate"]
            # Dedupe on coordinate so one doc per Type.field — the multiple
            # worlds produce many copies otherwise and they clutter the plot.
            if coord in seen:
                continue
            seen.add(coord)
            coords.append(coord)
            texts.append(c.get("retrieval_text") or c.get("field_semantic_text") or c.get("full_text") or coord)
    return coords, texts


def pick_demo_query(queries_path: Path, base_corpus, tuned_corpus, coords, base_model, tuned_model, limit_queries: int = 80) -> tuple[str, str]:
    """Fallback: find a query where tuned moved target from low rank → #1."""
    rows = []
    with queries_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit_queries:
                break
            rows.append(json.loads(line))
    coord_idx = {c: i for i, c in enumerate(coords)}
    queries = [r["query"] for r in rows]
    base_q = encode_with_resolution(base_model, queries, prompt_name="query")
    tuned_q = encode_with_resolution(tuned_model, queries, prompt_name="query")
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
        if tuned_rank != 0 or base_rank < 2:
            continue
        base_top1_coord = coords[int(np.argmax(base_scores))]
        same_field = base_top1_coord.split(".")[-1].lower() == target.split(".")[-1].lower()
        different_owner = base_top1_coord.split(".")[0] != target.split(".")[0]
        sibling_bonus = 10 if (same_field and different_owner) else 0
        score = base_rank + sibling_bonus
        if score > best_score:
            best_score = score
            best = (r["query"], target)
    if best is None:
        raise SystemExit("No clear winning example found; widen --limit-queries or pass --query/--target.")
    return best


def plot_embedding_space(query_emb_base, query_emb_tuned, coord_embs_base, coord_embs_tuned,
                         coords, target, siblings_idx, random_idx, out_path: Path) -> None:
    """Two-panel PCA of the same points embedded by base vs tuned model."""
    # Fit PCA separately per model so each panel uses its own principal axes —
    # otherwise the tuned panel looks weird since the manifold shifted.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, q_emb, coord_embs, title in [
        (axes[0], query_emb_base, coord_embs_base, "Base model (Qwen3-Embedding-0.6B)"),
        (axes[1], query_emb_tuned, coord_embs_tuned, "Fine-tuned model"),
    ]:
        combined = np.vstack([q_emb[None, :], coord_embs])
        pca = PCA(n_components=2, random_state=0)
        proj = pca.fit_transform(combined)
        q2d = proj[0]
        coord2d = proj[1:]

        # Background: random non-sibling coords, faded.
        ax.scatter(coord2d[random_idx, 0], coord2d[random_idx, 1],
                   s=14, c="#cccccc", alpha=0.5, label="other corpus", zorder=1)
        # Siblings (same field name): yellow.
        sibling_pts = coord2d[siblings_idx]
        ax.scatter(sibling_pts[:, 0], sibling_pts[:, 1],
                   s=80, c="#f4c542", edgecolors="#8a6a00", linewidth=0.6,
                   label="siblings (same field name)", zorder=3)
        # Target: green, large.
        target_idx_local = coords.index(target)
        tx, ty = coord2d[target_idx_local]
        ax.scatter(tx, ty, s=220, c="#2ecc71", edgecolors="black", linewidth=1.2,
                   label=f"target: {target}", zorder=5)
        # Label the target explicitly.
        ax.annotate(target.split(".")[0], (tx, ty), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, fontweight="bold",
                    color="#196f3d")
        # Query: blue star.
        ax.scatter(q2d[0], q2d[1], s=260, c="#3498db", marker="*",
                   edgecolors="black", linewidth=0.8, label="query", zorder=6)
        # Label the closest 3 siblings too (always the confusing ones).
        sibling_dists = np.linalg.norm(sibling_pts - q2d, axis=1)
        for rank, idx in enumerate(np.argsort(sibling_dists)[:3]):
            sx, sy = sibling_pts[idx]
            sc = coords[siblings_idx[idx]]
            ax.annotate(sc.split(".")[0], (sx, sy), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color="#8a6a00", alpha=0.85)

        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Query neighborhood — PCA of embeddings (target: {target})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_sibling_cosines(q_base, q_tuned, coord_embs_base, coord_embs_tuned,
                          coords, target, siblings_idx, out_path: Path,
                          query_text: str = "") -> None:
    """Bar chart: cosine(query, target + top-sibling distractors), base vs tuned."""
    # Rank siblings by baseline similarity so the worst baseline confusers
    # show up in the chart — that's what makes the flip story concrete.
    base_sims_all = coord_embs_base @ q_base
    tuned_sims_all = coord_embs_tuned @ q_tuned
    target_idx_local = coords.index(target)

    # Pick up to 6 top-baseline-similarity siblings (excluding target itself).
    sib_ranked = sorted(siblings_idx, key=lambda i: -base_sims_all[i])
    sib_ranked = [i for i in sib_ranked if coords[i] != target][:6]
    chosen = [target_idx_local] + sib_ranked
    labels = []
    for idx in chosen:
        c = coords[idx]
        owner, field = c.split(".", 1)
        labels.append(owner)  # field is shared, owner disambiguates

    base_bars = [float(base_sims_all[i]) for i in chosen]
    tuned_bars = [float(tuned_sims_all[i]) for i in chosen]

    x = np.arange(len(chosen))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(chosen) * 1.1), 5.2))
    ax.bar(x - width / 2, base_bars, width, label="base", color="#bfbfbf")
    bars_tuned = ax.bar(x + width / 2, tuned_bars, width, label="tuned", color="#3498db")

    # Highlight the target bars.
    bars_tuned[0].set_color("#2ecc71")
    bars_tuned[0].set_edgecolor("black")
    bars_tuned[0].set_linewidth(1)

    # Annotate values on top.
    for i, (b, t) in enumerate(zip(base_bars, tuned_bars)):
        ax.text(i - width / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8, color="#555")
        ax.text(i + width / 2, t + 0.005, f"{t:.2f}", ha="center", fontsize=8,
                color="#196f3d" if i == 0 else "#1f4e79", fontweight="bold" if i == 0 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel(f"cosine(query, <owner>.{target.split('.')[1]})")
    ax.set_title(f"Query → candidate similarity\n(target owner {target.split('.')[0]} highlighted in green)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")

    # Put the user query front-and-center so the plot is self-contained.
    if query_text:
        wrapped = _wrap(query_text, width=90)
        fig.suptitle(
            f'"{wrapped}"',
            fontsize=12, y=1.02, ha="center", fontstyle="italic",
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}")


def _wrap(text: str, width: int = 90) -> str:
    """Minimal word-wrap for matplotlib titles (no textwrap import cost)."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--tuned-model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", help="test.jsonl; only needed if --query/--target not supplied")
    ap.add_argument("--query", help="Explicit query text to visualize.")
    ap.add_argument("--target", help="Explicit target coordinate (Type.field).")
    ap.add_argument("--out-dir", default="artifacts/evals/plots")
    ap.add_argument("--random-non-siblings", type=int, default=200)
    ap.add_argument("--limit-queries", type=int, default=80, help="Scan window when auto-picking a demo query.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("loading corpus + encoding with both models...")
    coords, texts = load_corpus(Path(args.corpus))
    print(f"  {len(coords)} unique coordinates")

    coord_embs_base = encode_with_resolution(args.base_model, texts, prompt_name="document")
    coord_embs_tuned = encode_with_resolution(args.tuned_model, texts, prompt_name="document")

    # Resolve query + target.
    if args.query and args.target:
        query, target = args.query, args.target
    else:
        if not args.queries:
            raise SystemExit("Need --queries when --query/--target not given.")
        print("picking demo query automatically...")
        query, target = pick_demo_query(Path(args.queries), coord_embs_base, coord_embs_tuned,
                                         coords, args.base_model, args.tuned_model,
                                         limit_queries=args.limit_queries)
        print(f"  picked: target={target}")
        print(f"  query:  {query}")

    if target not in coords:
        raise SystemExit(f"target {target} not in deduped corpus; check --target spelling.")

    # Identify siblings = same field_name, different owner.
    target_field = target.split(".", 1)[1].lower()
    siblings_idx = [i for i, c in enumerate(coords)
                    if c != target and c.split(".", 1)[1].lower() == target_field]
    print(f"  found {len(siblings_idx)} siblings with field={target_field}")

    # Random non-siblings for the 2D background context.
    rng = np.random.default_rng(0)
    non_sibling_pool = [i for i, c in enumerate(coords)
                         if c != target and i not in siblings_idx]
    random_idx = rng.choice(non_sibling_pool, size=min(args.random_non_siblings, len(non_sibling_pool)),
                             replace=False).tolist()

    # Subset embeddings to what we'll render (plus target + siblings always in).
    keep_idx = sorted(set([coords.index(target)] + siblings_idx + random_idx))
    kept_coords = [coords[i] for i in keep_idx]
    kept_base = coord_embs_base[keep_idx]
    kept_tuned = coord_embs_tuned[keep_idx]
    local_sibling_idx = [kept_coords.index(coords[i]) for i in siblings_idx]
    local_random_idx = [kept_coords.index(coords[i]) for i in random_idx]

    query_emb_base = encode_with_resolution(args.base_model, [query], prompt_name="query")[0]
    query_emb_tuned = encode_with_resolution(args.tuned_model, [query], prompt_name="query")[0]

    plot_embedding_space(
        query_emb_base, query_emb_tuned, kept_base, kept_tuned,
        kept_coords, target, local_sibling_idx, local_random_idx,
        out / "embedding_space.png",
    )
    plot_sibling_cosines(
        query_emb_base, query_emb_tuned, kept_base, kept_tuned,
        kept_coords, target, local_sibling_idx,
        out / "sibling_cosines.png",
        query_text=query,
    )

    # Also dump the raw numbers so we can verify / splice into the README.
    base_target_sim = float(kept_base[kept_coords.index(target)] @ query_emb_base)
    tuned_target_sim = float(kept_tuned[kept_coords.index(target)] @ query_emb_tuned)
    base_target_rank = int(np.sum((coord_embs_base @ query_emb_base) > base_target_sim)) + 1
    tuned_target_rank = int(np.sum((coord_embs_tuned @ query_emb_tuned) > tuned_target_sim)) + 1
    print(json.dumps({
        "query": query,
        "target": target,
        "sibling_count": len(siblings_idx),
        "base_target_rank": base_target_rank,
        "tuned_target_rank": tuned_target_rank,
        "base_target_cosine": round(base_target_sim, 4),
        "tuned_target_cosine": round(tuned_target_sim, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
