from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from datasets import Dataset
except Exception:
    class Dataset(list):
        @classmethod
        def from_list(cls, rows):
            ds = cls(rows)
            ds.column_names = list(rows[0].keys()) if rows else []
            return ds

try:
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
except Exception:
    SentenceTransformer = None
    SentenceTransformerTrainer = None
    losses = None
    SentenceTransformerTrainingArguments = None

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.structural_views import get_positive_texts, normalize_primary_retrieval_view, parse_positive_views
from graphql_finetuning_pipeline.training.epoch_eval import EpochEvalCallback
from graphql_finetuning_pipeline.utils.io import ensure_dir


LOSS_CHOICES = {"mnrl", "cached_mnrl", "triplet"}


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    use_lora: bool = True
    tracking_backend: str = "none"
    experiment_name: str = "graphql-field-embedder-finetune"
    eval_every_epoch: bool = False
    positive_views: list[str] | None = None
    primary_retrieval_view: str = "semantic"
    loss: str = "cached_mnrl"
    num_hard_negatives: int = 4
    mnrl_scale: float = 20.0
    mnrl_mini_batch_size: int = 16
    precision: str = "auto"
    seed: int = 42


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
        return


def _pick_positive_view(coord: str, coord_to_docs: dict[str, list[str]], primary_view_docs: dict[str, str], rng: random.Random) -> str:
    docs = coord_to_docs.get(coord)
    if not docs:
        return primary_view_docs[coord]
    return rng.choice(docs)


def _build_triplet_rows(
    rows: list[QueryRecord],
    coord_to_docs: dict[str, list[str]],
    primary_view_docs: dict[str, str],
    rng: random.Random,
) -> list[dict[str, str]]:
    all_coords = list(primary_view_docs.keys())
    triplets: list[dict[str, str]] = []
    for row in rows:
        positives = [p for p in (row.relevant_coordinates or [row.positive_coordinate]) if p in coord_to_docs]
        if not positives:
            continue
        negatives = [n for n in row.negative_coordinates if n in primary_view_docs and n not in positives]
        if not negatives:
            negatives = [c for c in all_coords if c not in positives][:1]
        if not negatives:
            continue
        pos = positives[0]
        triplets.append({"anchor": row.query, "positive": _pick_positive_view(pos, coord_to_docs, primary_view_docs, rng), "negative": primary_view_docs[rng.choice(negatives)]})
    return triplets


def _build_multineg_rows(
    rows: list[QueryRecord],
    coord_to_docs: dict[str, list[str]],
    primary_view_docs: dict[str, str],
    num_hard_negatives: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    """Builds rows of shape (anchor, positive, negative_1, ..., negative_N).

    Every row has the same number of negative columns so sentence-transformers
    can treat them as a fixed-width tuple for MultipleNegativesRankingLoss /
    CachedMultipleNegativesRankingLoss. Missing hard negatives are backfilled
    with random in-corpus negatives from outside the relevant set.
    """
    if num_hard_negatives < 1:
        raise ValueError("num_hard_negatives must be >= 1 for multi-negative losses")
    all_coords = list(primary_view_docs.keys())
    out: list[dict[str, str]] = []
    for row in rows:
        positives = [p for p in (row.relevant_coordinates or [row.positive_coordinate]) if p in coord_to_docs]
        if not positives:
            continue
        pos = positives[0]
        excluded = set(positives)
        ordered_negs = [n for n in row.negative_coordinates if n in primary_view_docs and n not in excluded]
        # De-duplicate while preserving ranked order from the dataset builder.
        seen: set[str] = set()
        ranked_negs: list[str] = []
        for n in ordered_negs:
            if n in seen:
                continue
            seen.add(n)
            ranked_negs.append(n)
        if len(ranked_negs) < num_hard_negatives:
            pool = [c for c in all_coords if c not in excluded and c not in seen]
            rng.shuffle(pool)
            ranked_negs.extend(pool[: num_hard_negatives - len(ranked_negs)])
        negs = ranked_negs[:num_hard_negatives]
        if len(negs) < num_hard_negatives:
            continue
        record = {
            "anchor": row.query,
            "positive": _pick_positive_view(pos, coord_to_docs, primary_view_docs, rng),
        }
        for i, n in enumerate(negs, start=1):
            record[f"negative_{i}"] = primary_view_docs[n]
        out.append(record)
    return out


def build_pair_dataset(
    rows: list[QueryRecord],
    coord_to_docs: dict[str, list[str]],
    primary_view_docs: dict[str, str],
    *,
    loss: str = "triplet",
    num_hard_negatives: int = 4,
    seed: int = 42,
) -> Dataset:
    rng = random.Random(seed)
    loss_key = loss.lower()
    if loss_key == "triplet":
        return Dataset.from_list(_build_triplet_rows(rows, coord_to_docs, primary_view_docs, rng))
    if loss_key in {"mnrl", "cached_mnrl"}:
        return Dataset.from_list(_build_multineg_rows(rows, coord_to_docs, primary_view_docs, num_hard_negatives, rng))
    raise ValueError(f"Unknown loss '{loss}'. Valid: {sorted(LOSS_CHOICES)}")


def compute_warmup_steps(train_size: int, batch_size: int, epochs: int, warmup_ratio: float = 0.05) -> int:
    steps_per_epoch = max(math.ceil(train_size / max(batch_size, 1)), 1)
    total_steps = max(steps_per_epoch * max(epochs, 1), 1)
    return max(int(total_steps * warmup_ratio), 1)


def _resolve_precision(precision: str) -> tuple[bool, bool]:
    p = (precision or "auto").lower()
    if p == "fp16":
        return False, True
    if p == "bf16":
        return True, False
    if p == "fp32" or p == "none":
        return False, False
    # auto: prefer bf16 on CUDA if supported, else fp16 on CUDA, else fp32
    try:
        import torch

        if torch.cuda.is_available():
            if getattr(torch.cuda, "is_bf16_supported", lambda: False)():
                return True, False
            return False, True
    except Exception:
        pass
    return False, False


def _build_loss(loss_name: str, model: SentenceTransformer, cfg: TrainConfig):
    loss_name = loss_name.lower()
    if loss_name == "triplet":
        return losses.TripletLoss(model=model)
    if loss_name == "mnrl":
        return losses.MultipleNegativesRankingLoss(model=model, scale=cfg.mnrl_scale)
    if loss_name == "cached_mnrl":
        cached = getattr(losses, "CachedMultipleNegativesRankingLoss", None)
        if cached is None:
            return losses.MultipleNegativesRankingLoss(model=model, scale=cfg.mnrl_scale)
        return cached(model=model, scale=cfg.mnrl_scale, mini_batch_size=cfg.mnrl_mini_batch_size)
    raise ValueError(f"Unknown loss '{loss_name}'. Valid: {sorted(LOSS_CHOICES)}")


def train_biencoder(
    train_rows: list[QueryRecord],
    val_rows: list[QueryRecord],
    corpus_rows: list[CorpusRecord],
    out_dir: Path,
    cfg: TrainConfig,
    corpus_hash: str | None = None,
    benchmark_sets: dict[str, list[QueryRecord]] | None = None,
) -> dict:
    if SentenceTransformer is None or SentenceTransformerTrainer is None or losses is None or SentenceTransformerTrainingArguments is None:
        raise ImportError("sentence-transformers is required for training")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    model = SentenceTransformer(cfg.model_name)
    model.max_seq_length = cfg.max_seq_length
    if cfg.use_lora:
        _maybe_attach_lora(model)

    positive_views = parse_positive_views(",".join(cfg.positive_views) if cfg.positive_views else None)
    primary_retrieval_view = normalize_primary_retrieval_view(cfg.primary_retrieval_view)

    coord_to_docs: dict[str, list[str]] = {}
    primary_view_docs: dict[str, str] = {}
    for c in corpus_rows:
        docs = get_positive_texts(c, positive_views)
        if docs:
            coord_to_docs[c.coordinate] = docs
            primary_view_docs[c.coordinate] = next((d for d in docs if d == c.retrieval_text), docs[0])

    if not coord_to_docs:
        raise ValueError("No structural corpus texts available for selected positive views.")

    train_dataset = build_pair_dataset(
        train_rows,
        coord_to_docs,
        primary_view_docs,
        loss=cfg.loss,
        num_hard_negatives=cfg.num_hard_negatives,
        seed=cfg.seed,
    )
    val_dataset = build_pair_dataset(
        val_rows,
        coord_to_docs,
        primary_view_docs,
        loss=cfg.loss,
        num_hard_negatives=cfg.num_hard_negatives,
        seed=cfg.seed + 1,
    )
    if len(train_dataset) == 0:
        raise ValueError("No train rows available after mapping query targets to corpus docs.")

    loss = _build_loss(cfg.loss, model, cfg)
    warmup_steps = compute_warmup_steps(len(train_dataset), cfg.batch_size, cfg.epochs)
    bf16, fp16 = _resolve_precision(cfg.precision)

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
        fp16=fp16,
        bf16=bf16,
        dataloader_drop_last=cfg.loss != "triplet",
        report_to=[] if cfg.tracking_backend == "none" else [cfg.tracking_backend],
        run_name=cfg.experiment_name,
        seed=cfg.seed,
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

    if cfg.use_lora:
        try:
            first = model._first_module()
            auto_model = first.auto_model
            if hasattr(auto_model, "merge_and_unload"):
                first.auto_model = auto_model.merge_and_unload()
        except Exception:
            pass

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
        "loss": cfg.loss,
        "num_hard_negatives": cfg.num_hard_negatives if cfg.loss != "triplet" else 1,
        "mnrl_scale": cfg.mnrl_scale if cfg.loss != "triplet" else None,
        "mnrl_mini_batch_size": cfg.mnrl_mini_batch_size if cfg.loss == "cached_mnrl" else None,
        "precision": {"bf16": bf16, "fp16": fp16},
        "objective": "multi-negative-ranking-field-coordinate-retrieval" if cfg.loss != "triplet" else "triplet-field-coordinate-retrieval",
    }
    (out_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
