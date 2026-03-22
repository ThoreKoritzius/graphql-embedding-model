from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.structural_views import (
    get_positive_texts,
    normalize_primary_retrieval_view,
    parse_positive_views,
)
from graphql_finetuning_pipeline.training.epoch_eval import EpochEvalCallback
from graphql_finetuning_pipeline.utils.io import ensure_dir


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    use_lora: bool = True
    tracking_backend: str = "none"
    experiment_name: str = "graphql-embedder-finetune"
    eval_every_epoch: bool = False
    positive_views: list[str] | None = None
    primary_retrieval_view: str = "sdl"


def _maybe_attach_lora(model: SentenceTransformer, rank: int = 16) -> None:
    try:
        from peft import LoraConfig, TaskType, get_peft_model

        first = model._first_module()
        auto_model = first.auto_model
        cfg = LoraConfig(
            r=rank,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        first.auto_model = get_peft_model(auto_model, cfg)
    except Exception:
        # Keep training functional even if LoRA cannot attach for a given architecture.
        return


def _build_pair_rows(rows: list[QueryRecord], type_to_docs: dict[str, list[str]]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for row in rows:
        primary = row.primary_type_id or row.target_type_id
        positives = row.relevant_type_ids or [primary]
        positives = [p for p in positives if p in type_to_docs]
        if primary in type_to_docs and primary not in positives:
            positives.insert(0, primary)
        if not positives:
            continue
        for p in positives:
            for doc in type_to_docs[p]:
                pairs.append({"anchor": row.query, "positive": doc})
    return pairs


def build_pair_dataset(rows: list[QueryRecord], type_to_docs: dict[str, list[str]]) -> Dataset:
    return Dataset.from_list(_build_pair_rows(rows, type_to_docs))


def compute_warmup_steps(train_size: int, batch_size: int, epochs: int, warmup_ratio: float = 0.05) -> int:
    steps_per_epoch = max(math.ceil(train_size / max(batch_size, 1)), 1)
    total_steps = max(steps_per_epoch * max(epochs, 1), 1)
    return max(int(total_steps * warmup_ratio), 1)


def train_biencoder(
    train_rows: list[QueryRecord],
    val_rows: list[QueryRecord],
    corpus_rows: list[CorpusRecord],
    out_dir: Path,
    cfg: TrainConfig,
    corpus_hash: str | None = None,
    benchmark_sets: dict[str, list[QueryRecord]] | None = None,
) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    model = SentenceTransformer(cfg.model_name)
    model.max_seq_length = cfg.max_seq_length

    if cfg.use_lora:
        _maybe_attach_lora(model)

    positive_views = parse_positive_views(",".join(cfg.positive_views) if cfg.positive_views else None)
    primary_retrieval_view = normalize_primary_retrieval_view(cfg.primary_retrieval_view)

    type_to_docs: dict[str, list[str]] = {}
    for c in corpus_rows:
        docs = get_positive_texts(c, positive_views)
        if docs:
            type_to_docs[c.type_id] = docs

    if not type_to_docs:
        raise ValueError(
            "No structural corpus texts available for selected positive views. "
            "Run `graphft backfill-structural-corpus` and use the produced corpus file."
        )

    train_dataset = build_pair_dataset(train_rows, type_to_docs)
    val_dataset = build_pair_dataset(val_rows, type_to_docs)

    if len(train_dataset) == 0:
        raise ValueError(
            "No train pairs available after mapping query targets to corpus docs. "
            "Check that train split target_type_id values exist in corpus."
        )

    loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = compute_warmup_steps(len(train_dataset), cfg.batch_size, cfg.epochs)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        eval_strategy="steps" if len(val_dataset) > 0 else "no",
        eval_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=20,
        warmup_steps=warmup_steps,
        fp16=False,
        bf16=False,
        report_to=[] if cfg.tracking_backend == "none" else [cfg.tracking_backend],
        run_name=cfg.experiment_name,
    )

    if cfg.tracking_backend == "mlflow":
        try:
            import mlflow

            mlflow.set_experiment(cfg.experiment_name)
        except Exception:
            pass

    callbacks: list[Any] = []
    if cfg.eval_every_epoch and benchmark_sets:
        callbacks.append(
            EpochEvalCallback(
                model=model,
                corpus_rows=corpus_rows,
                benchmark_sets=benchmark_sets,
                out_dir=out_dir,
                tracking_backend=cfg.tracking_backend,
                retrieval_view=primary_retrieval_view,
            )
        )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        loss=loss,
        callbacks=callbacks,
    )
    trainer.train()

    # If PEFT adapters are attached, try to merge for a self-contained saved model.
    if cfg.use_lora:
        try:
            first = model._first_module()
            auto_model = first.auto_model
            if hasattr(auto_model, "merge_and_unload"):
                first.auto_model = auto_model.merge_and_unload()
        except Exception:
            pass

    # Save directly to out_dir so this path can be passed to eval/index commands.
    model.save(str(out_dir))

    manifest = {
        "model_name": cfg.model_name,
        "model_dir": str(out_dir.resolve()),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "max_seq_length": cfg.max_seq_length,
        "use_lora": cfg.use_lora,
        "normalize_embeddings": True,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "warmup_steps": warmup_steps,
        "tracking_backend": cfg.tracking_backend,
        "experiment_name": cfg.experiment_name,
        "corpus_hash": corpus_hash,
        "eval_every_epoch": cfg.eval_every_epoch,
        "positive_views": positive_views,
        "primary_retrieval_view": primary_retrieval_view,
        "structural_corpus": True,
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if cfg.tracking_backend == "wandb":
        try:
            import wandb

            wandb.config.update(
                {
                    "model_name": cfg.model_name,
                    "epochs": cfg.epochs,
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "max_seq_length": cfg.max_seq_length,
                    "use_lora": cfg.use_lora,
                    "corpus_hash": corpus_hash,
                    "positive_views": positive_views,
                    "primary_retrieval_view": primary_retrieval_view,
                },
                allow_val_change=True,
            )
            art = wandb.Artifact("trained-model-manifest", type="metadata")
            art.add_file(str(out_dir / "training_manifest.json"))
            wandb.log_artifact(art)
        except Exception:
            pass

    return manifest
