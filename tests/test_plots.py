import json
from pathlib import Path

from graphql_finetuning_pipeline.eval import plots


class _FakePLT:
    def figure(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None

    def xlabel(self, *args, **kwargs):
        return None

    def ylabel(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def tight_layout(self):
        return None

    def savefig(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PNG")

    def close(self):
        return None

    def bar(self, *args, **kwargs):
        return None

    def xticks(self, *args, **kwargs):
        return None

    def yticks(self, *args, **kwargs):
        return None

    def imshow(self, *args, **kwargs):
        return None

    def colorbar(self, *args, **kwargs):
        return None


def test_plot_serialization_outputs_csv_and_png(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(plots, "_require_matplotlib", lambda: _FakePLT())

    epoch_path = tmp_path / "epoch_metrics.jsonl"
    epoch_path.write_text(
        "\n".join(
            [
                json.dumps({"epoch": 1, "benchmark": "synthetic_holdout", "recall@5": 0.5, "mrr@10": 0.4, "ndcg@10": 0.45}),
                json.dumps({"epoch": 2, "benchmark": "synthetic_holdout", "recall@5": 0.6, "mrr@10": 0.5, "ndcg@10": 0.55}),
            ]
        ),
        encoding="utf-8",
    )

    bench_path = tmp_path / "benchmarks_summary.json"
    bench_path.write_text(
        json.dumps(
            {
                "synthetic_holdout": {"recall@5": 0.6, "mrr@10": 0.5, "ndcg@10": 0.55},
                "realism_eval": {"recall@5": 0.5, "mrr@10": 0.4, "ndcg@10": 0.43},
            }
        ),
        encoding="utf-8",
    )

    out = plots.plot_epoch_metrics(epoch_path, tmp_path)
    out2 = plots.plot_benchmark_comparison(bench_path, tmp_path)

    assert Path(out["epoch_metrics_csv"]).exists()
    assert Path(out["metric_vs_epoch_png"]).exists()
    assert Path(out2["benchmark_csv"]).exists()
    assert Path(out2["benchmark_bar_png"]).exists()
    assert Path(out2["slice_heatmap_png"]).exists()
