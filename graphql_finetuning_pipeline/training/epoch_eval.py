from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback
except Exception:
    class TrainerCallback:
        pass

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.eval.benchmark import evaluate_benchmark_set
from graphql_finetuning_pipeline.utils.io import ensure_dir


BEST_METRIC_CHOICES = {"ndcg@10", "mrr@10", "recall@1", "recall@3", "recall@5", "recall@10", "exact_match@1"}


class EpochEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        model: Any,
        corpus_rows: list[CorpusRecord],
        benchmark_sets: dict[str, list[QueryRecord]],
        out_dir: Path,
        tracking_backend: str = "none",
        retrieval_view: str = "semantic",
        best_metric: str = "ndcg@10",
        best_benchmark: str | None = None,
    ) -> None:
        self.model = model
        self.corpus_rows = corpus_rows
        self.benchmark_sets = benchmark_sets
        self.out_dir = out_dir
        self.metrics_dir = out_dir / "metrics"
        ensure_dir(self.metrics_dir)
        self.metrics_path = self.metrics_dir / "epoch_metrics.jsonl"
        self.tracking_backend = tracking_backend
        self.retrieval_view = retrieval_view
        if best_metric not in BEST_METRIC_CHOICES:
            raise ValueError(f"best_metric must be one of {sorted(BEST_METRIC_CHOICES)}; got {best_metric!r}")
        self.best_metric = best_metric
        self.best_benchmark = best_benchmark
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.best_dir = out_dir / "best"

    def _log_wandb(self, payload: dict) -> None:
        if self.tracking_backend != "wandb":
            return
        try:
            import wandb

            wandb.log(payload)
        except Exception:
            return

    def _resolve_primary_benchmark(self) -> str | None:
        if self.best_benchmark and self.best_benchmark in self.benchmark_sets:
            return self.best_benchmark
        if not self.benchmark_sets:
            return None
        return next(iter(self.benchmark_sets.keys()))

    def _maybe_promote_best(self, epoch: int, records: list[dict]) -> None:
        primary = self._resolve_primary_benchmark()
        if primary is None:
            return
        target = next((r for r in records if r["benchmark"] == primary), None)
        if target is None:
            return
        score = target.get(self.best_metric)
        if score is None:
            return
        if self.best_score is not None and score <= self.best_score:
            return
        self.best_score = float(score)
        self.best_epoch = epoch
        ensure_dir(self.best_dir)
        self.model.save(str(self.best_dir))
        info = {
            "epoch": epoch,
            "benchmark": primary,
            "metric": self.best_metric,
            "score": float(score),
        }
        (self.best_dir / "epoch_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        records = []
        for name, rows in self.benchmark_sets.items():
            result = evaluate_benchmark_set(rows, self.corpus_rows, self.model, retrieval_view=self.retrieval_view)
            rec = {
                "epoch": epoch,
                "benchmark": name,
                "recall@1": result["recall@1"],
                "recall@5": result["recall@5"],
                "recall@10": result["recall@10"],
                "mrr@10": result["mrr@10"],
                "ndcg@10": result["ndcg@10"],
                "exact_match@1": result.get("exact_match@1", 0.0),
                "recall@3": result.get("recall@3", 0.0),
                "same_owner_wrong_field_rate@1": result.get("same_owner_wrong_field_rate@1", 0.0),
            }
            records.append(rec)
            self._log_wandb({f"epoch/{name}/recall@5": rec["recall@5"], "epoch": epoch})
            self._log_wandb({f"epoch/{name}/mrr@10": rec["mrr@10"], "epoch": epoch})
            self._log_wandb({f"epoch/{name}/ndcg@10": rec["ndcg@10"], "epoch": epoch})

            if self.tracking_backend == "wandb":
                try:
                    import wandb

                    table_rows = result["per_query"][:150]
                    if table_rows:
                        cols = ["query_id", "query", "positive_coordinate", "intent", "difficulty", "source", "top_k", "exact_match@1", "recall@5", "mrr@10", "ndcg@10"]
                        table = wandb.Table(columns=cols)
                        for row in table_rows:
                            table.add_data(
                                row.get("query_id"),
                                row.get("query"),
                                row.get("positive_coordinate"),
                                row.get("intent"),
                                row.get("difficulty"),
                                row.get("source"),
                                ",".join(row.get("top_k", [])),
                                row.get("exact_match@1"),
                                row.get("recall@5"),
                                row.get("mrr@10"),
                                row.get("ndcg@10"),
                            )
                        wandb.log({f"epoch/{name}/predictions": table, "epoch": epoch})
                except Exception:
                    pass

        with self.metrics_path.open("a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        self._maybe_promote_best(epoch, records)

        return control
