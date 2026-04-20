"""Plot baseline vs tuned metrics across one or more eval runs.

Usage:
  python tools/plot_eval.py artifacts/evals/qwen3-v7 artifacts/evals/qwen3-v7-e3 \
      --out artifacts/evals/plots

Each argument is a directory containing baseline_metrics.json + tuned_metrics.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = ["exact_match@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]


def load_run(run_dir: Path) -> tuple[dict, dict]:
    base = json.loads((run_dir / "baseline_metrics.json").read_text())
    tuned = json.loads((run_dir / "tuned_metrics.json").read_text())
    return base, tuned


def plot_bars(runs: list[tuple[str, dict, dict]], out: Path) -> None:
    """Grouped bar chart: one group per metric, one bar per (run, split)."""
    fig, ax = plt.subplots(figsize=(max(10, 2 * len(runs)), 5.5))
    n_groups = len(METRICS)
    width = 0.8 / (len(runs) * 2)
    x = np.arange(n_groups)

    palette = plt.colormaps["tab20"].colors
    for ri, (name, base, tuned) in enumerate(runs):
        b_vals = [base.get(m, 0.0) for m in METRICS]
        t_vals = [tuned.get(m, 0.0) for m in METRICS]
        offset_base = (2 * ri - len(runs) + 0.5) * width
        offset_tuned = (2 * ri - len(runs) + 1.5) * width
        ax.bar(x + offset_base, b_vals, width, label=f"{name} / baseline", color=palette[(ri * 2) % len(palette)], alpha=0.6)
        ax.bar(x + offset_tuned, t_vals, width, label=f"{name} / tuned", color=palette[(ri * 2) % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=20)
    ax.set_ylabel("score")
    ax.set_title("Retrieval metrics — baseline vs tuned")
    ax.set_ylim(0, max(0.5, max(max(base.get(m, 0.0) for m in METRICS) for _, base, _ in runs) * 1.15,
                       max(max(t.get(m, 0.0) for m in METRICS) for _, _, t in runs) * 1.15))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "metrics_bars.png", dpi=150)
    print(f"wrote {out / 'metrics_bars.png'}")


def plot_deltas(runs: list[tuple[str, dict, dict]], out: Path) -> None:
    """Lift over baseline, per metric per run. Positive = tuned wins."""
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(runs)), 5))
    width = 0.8 / max(1, len(runs))
    x = np.arange(len(METRICS))
    palette = plt.colormaps["tab10"].colors

    for ri, (name, base, tuned) in enumerate(runs):
        deltas = [tuned.get(m, 0.0) - base.get(m, 0.0) for m in METRICS]
        offset = (ri - (len(runs) - 1) / 2) * width
        bars = ax.bar(x + offset, deltas, width, label=name, color=palette[ri % len(palette)])
        for bar, d in zip(bars, deltas):
            ax.annotate(f"+{d:.3f}" if d >= 0 else f"{d:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, d),
                        xytext=(0, 3 if d >= 0 else -10), textcoords="offset points",
                        ha="center", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, rotation=20)
    ax.set_ylabel("tuned − baseline")
    ax.set_title("Lift over baseline (higher = fine-tune helps)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "metrics_deltas.png", dpi=150)
    print(f"wrote {out / 'metrics_deltas.png'}")


def plot_recall_at_k(runs: list[tuple[str, dict, dict]], out: Path) -> None:
    """Recall@K curve for each run."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ks = [1, 3, 5, 10]
    palette = plt.colormaps["tab10"].colors
    for ri, (name, base, tuned) in enumerate(runs):
        c = palette[ri % len(palette)]
        b_vals = [base.get(f"recall@{k}", 0.0) for k in ks]
        t_vals = [tuned.get(f"recall@{k}", 0.0) for k in ks]
        ax.plot(ks, b_vals, marker="o", linestyle="--", color=c, alpha=0.6, label=f"{name} / baseline")
        ax.plot(ks, t_vals, marker="o", linestyle="-", color=c, label=f"{name} / tuned")
    ax.set_xticks(ks)
    ax.set_xlabel("K")
    ax.set_ylabel("recall@K")
    ax.set_title("Recall @ K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "recall_at_k.png", dpi=150)
    print(f"wrote {out / 'recall_at_k.png'}")


def print_table(runs: list[tuple[str, dict, dict]]) -> None:
    col_w = 24
    header = "metric".ljust(16)
    for name, *_ in runs:
        header += f"| {name} base".ljust(col_w) + f"| {name} tuned".ljust(col_w) + f"| delta".ljust(14)
    print(header)
    print("-" * len(header))
    for m in METRICS:
        row = m.ljust(16)
        for _, base, tuned in runs:
            b = base.get(m, 0.0)
            t = tuned.get(m, 0.0)
            row += f"| {b:.4f}".ljust(col_w) + f"| {t:.4f}".ljust(col_w) + f"| {t - b:+.4f}".ljust(14)
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="Paths to eval directories (each must contain baseline_metrics.json and tuned_metrics.json).")
    ap.add_argument("--out", default="artifacts/evals/plots", help="Directory to write PNGs into.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for rd in args.run_dirs:
        p = Path(rd)
        if not (p / "baseline_metrics.json").exists() or not (p / "tuned_metrics.json").exists():
            print(f"skip {p}: missing baseline_metrics.json or tuned_metrics.json")
            continue
        base, tuned = load_run(p)
        runs.append((p.name, base, tuned))

    if not runs:
        print("no usable runs found")
        return

    print_table(runs)
    plot_bars(runs, out)
    plot_deltas(runs, out)
    plot_recall_at_k(runs, out)


if __name__ == "__main__":
    main()
