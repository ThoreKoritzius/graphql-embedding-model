from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from graphql_finetuning_pipeline.utils.io import ensure_dir


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plot generation; install it in your environment") from exc


def plot_epoch_metrics(epoch_metrics_path: Path, out_dir: Path) -> dict[str, str]:
    ensure_dir(out_dir)
    rows = _load_jsonl(epoch_metrics_path)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[r["benchmark"]].append(r)

    csv_path = out_dir / "epoch_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "benchmark",
                "recall@5",
                "mrr@10",
                "ndcg@10",
                "set_recall_any@5",
                "coverage@10",
                "pair_recall@10",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "epoch": r["epoch"],
                    "benchmark": r["benchmark"],
                    "recall@5": r["recall@5"],
                    "mrr@10": r["mrr@10"],
                    "ndcg@10": r["ndcg@10"],
                    "set_recall_any@5": r.get("set_recall_any@5", 0.0),
                    "coverage@10": r.get("coverage@10", 0.0),
                    "pair_recall@10": r.get("pair_recall@10", 0.0),
                }
            )

    plt = _require_matplotlib()

    metric_plot = out_dir / "metric_vs_epoch.png"
    plt.figure(figsize=(10, 6))
    for bench, vals in grouped.items():
        vals = sorted(vals, key=lambda x: x["epoch"])
        xs = [v["epoch"] for v in vals]
        ys = [v["recall@5"] for v in vals]
        plt.plot(xs, ys, marker="o", label=f"{bench} recall@5")
    plt.xlabel("Epoch")
    plt.ylabel("Recall@5")
    plt.title("Recall@5 by Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metric_plot)
    plt.close()

    return {
        "epoch_metrics_csv": str(csv_path),
        "metric_vs_epoch_png": str(metric_plot),
    }


def plot_benchmark_comparison(benchmark_summary_path: Path, out_dir: Path) -> dict[str, str]:
    ensure_dir(out_dir)
    summary = json.loads(benchmark_summary_path.read_text(encoding="utf-8"))

    benches = sorted(summary.keys())
    recalls = [summary[b]["recall@5"] for b in benches]
    mrrs = [summary[b]["mrr@10"] for b in benches]
    transfer_gap = [summary[b].get("transfer_gap", {}).get("seen_vs_unseen_recall@5_gap", 0.0) for b in benches]

    csv_path = out_dir / "benchmark_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "recall@5", "mrr@10", "ndcg@10"])
        writer.writeheader()
        for b in benches:
            writer.writerow(
                {
                    "benchmark": b,
                    "recall@5": summary[b]["recall@5"],
                    "mrr@10": summary[b]["mrr@10"],
                    "ndcg@10": summary[b]["ndcg@10"],
                }
            )

    plt = _require_matplotlib()

    bar_plot = out_dir / "benchmark_bar.png"
    plt.figure(figsize=(10, 6))
    x = list(range(len(benches)))
    plt.bar([i - 0.15 for i in x], recalls, width=0.3, label="recall@5")
    plt.bar([i + 0.15 for i in x], mrrs, width=0.3, label="mrr@10")
    plt.xticks(x, benches, rotation=20)
    plt.ylabel("Score")
    plt.title("Benchmark Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(bar_plot)
    plt.close()

    heatmap_plot = out_dir / "slice_heatmap.png"
    # Basic heatmap from benchmark recall/mrr/ndcg.
    heat = [[summary[b]["recall@5"], summary[b]["mrr@10"], summary[b]["ndcg@10"]] for b in benches]
    plt.figure(figsize=(8, 5))
    plt.imshow(heat, aspect="auto")
    plt.yticks(range(len(benches)), benches)
    plt.xticks(range(3), ["recall@5", "mrr@10", "ndcg@10"])
    plt.colorbar()
    plt.title("Benchmark Score Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_plot)
    plt.close()

    transfer_plot = out_dir / "transfer_gap.png"
    plt.figure(figsize=(10, 5))
    x = list(range(len(benches)))
    plt.bar(x, transfer_gap)
    plt.xticks(x, benches, rotation=20)
    plt.ylabel("Seen-Unseen Recall@5 Gap")
    plt.title("World Transfer Gap by Benchmark")
    plt.tight_layout()
    plt.savefig(transfer_plot)
    plt.close()

    return {
        "benchmark_csv": str(csv_path),
        "benchmark_bar_png": str(bar_plot),
        "slice_heatmap_png": str(heatmap_plot),
        "transfer_gap_png": str(transfer_plot),
    }
