from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.eval.benchmark import evaluate_benchmark_set
from graphql_finetuning_pipeline.utils.io import ensure_dir


class EpochEvalCallback(TrainerCallback):
    def __init__(
        self,
        *,
        model: Any,
        corpus_rows: list[CorpusRecord],
        benchmark_sets: dict[str, list[QueryRecord]],
        out_dir: Path,
        tracking_backend: str = "none",
    ) -> None:
        self.model = model
        self.corpus_rows = corpus_rows
        self.benchmark_sets = benchmark_sets
        self.metrics_dir = out_dir / "metrics"
        ensure_dir(self.metrics_dir)
        self.metrics_path = self.metrics_dir / "epoch_metrics.jsonl"
        self.tracking_backend = tracking_backend

    def _log_wandb(self, payload: dict) -> None:
        if self.tracking_backend != "wandb":
            return
        try:
            import wandb

            wandb.log(payload)
        except Exception:
            return

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch or 0)
        records = []
        for name, rows in self.benchmark_sets.items():
            result = evaluate_benchmark_set(rows, self.corpus_rows, self.model)
            rec = {
                "epoch": epoch,
                "benchmark": name,
                "recall@1": result["recall@1"],
                "recall@5": result["recall@5"],
                "recall@10": result["recall@10"],
                "mrr@10": result["mrr@10"],
                "ndcg@10": result["ndcg@10"],
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
                        cols = ["query_id", "query", "target_type_id", "intent", "difficulty", "source", "top_k", "recall@5", "mrr@10", "ndcg@10"]
                        table = wandb.Table(columns=cols)
                        for row in table_rows:
                            table.add_data(
                                row.get("query_id"),
                                row.get("query"),
                                row.get("target_type_id"),
                                row.get("intent"),
                                row.get("difficulty"),
                                row.get("source"),
                                ",".join(row.get("top_k", [])),
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

        return control
