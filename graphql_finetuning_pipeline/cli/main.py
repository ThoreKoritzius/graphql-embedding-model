from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.models import CorpusRecord, GraphQLTypeRecord, QueryRecord
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.synthetic import (
    bootstrap_queries,
    build_benchmark_sets,
    expand_queries,
    quality_filter,
    split_queries,
)
from graphql_finetuning_pipeline.eval.retrieval_eval import evaluate
from graphql_finetuning_pipeline.retrieval.hard_negatives import mine_hard_negatives
from graphql_finetuning_pipeline.retrieval.index import build_index
from graphql_finetuning_pipeline.utils.io import ensure_dir, read_jsonl, sha256_file, write_json, write_jsonl


def _load_types(path: Path) -> list[GraphQLTypeRecord]:
    return [GraphQLTypeRecord.model_validate(x) for x in read_jsonl(path)]


def _load_corpus(path: Path) -> list[CorpusRecord]:
    return [CorpusRecord.model_validate(x) for x in read_jsonl(path)]


def _load_queries(path: Path) -> list[QueryRecord]:
    return [QueryRecord.model_validate(x) for x in read_jsonl(path)]


def cmd_ingest_schema(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    records, _ = parse_schema(Path(args.input))
    normalized_path = out_dir / "normalized_types.jsonl"
    write_jsonl(normalized_path, [r.model_dump() for r in records])

    meta = {
        "schema_hash": sha256_file(Path(args.input)),
        "schema_path": str(Path(args.input).resolve()),
        "type_count": len(records),
    }
    write_json(out_dir / "metadata" / "schema_hash.json", meta)
    print(json.dumps({"normalized": str(normalized_path), **meta}, indent=2))


def cmd_build_corpus(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "corpus")

    records = _load_types(Path(args.normalized))
    corpus_rows = build_corpus(records)
    path = out_dir / "corpus" / f"types_v{args.version}.jsonl"
    write_jsonl(path, [r.model_dump() for r in corpus_rows])

    print(json.dumps({"corpus": str(path), "count": len(corpus_rows)}, indent=2))


def cmd_generate_synthetic(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "splits")
    ensure_dir(out_dir / "benchmarks")

    corpus = _load_corpus(Path(args.corpus))
    seed = bootstrap_queries(corpus, rng_seed=args.seed)
    expanded = expand_queries(seed, max_expansions_per_seed=args.max_expansions)
    filtered = quality_filter(expanded, type_names={c.type_name for c in corpus})
    split_rows = split_queries(filtered, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)

    for split in ("train", "val", "test"):
        rows = [r.model_dump() for r in split_rows if r.split == split]
        write_jsonl(out_dir / "splits" / f"{split}.jsonl", rows)

    benchmarks = build_benchmark_sets(split_rows, corpus)
    for name, rows in benchmarks.items():
        write_jsonl(out_dir / "benchmarks" / f"{name}.jsonl", [r.model_dump() for r in rows])

    print(
        json.dumps(
            {
                "seed_count": len(seed),
                "expanded_count": len(expanded),
                "filtered_count": len(filtered),
                "train": len([r for r in split_rows if r.split == "train"]),
                "val": len([r for r in split_rows if r.split == "val"]),
                "test": len([r for r in split_rows if r.split == "test"]),
                "benchmarks": {k: len(v) for k, v in benchmarks.items()},
            },
            indent=2,
        )
    )


def cmd_mine_hard_negatives(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "splits")

    corpus = _load_corpus(Path(args.corpus))
    rows = _load_queries(Path(args.queries))
    updated = mine_hard_negatives(corpus, rows, hard_k=args.hard_k, easy_k=args.easy_k, medium_k=args.medium_k)

    target = out_dir / "splits" / Path(args.queries).name
    write_jsonl(target, [x.model_dump() for x in updated])
    print(json.dumps({"input": len(rows), "output": len(updated), "path": str(target)}, indent=2))


def cmd_train_embedder(args: argparse.Namespace) -> None:
    from graphql_finetuning_pipeline.training.train_embedder import TrainConfig, train_biencoder

    train_rows = _load_queries(Path(args.train))
    val_rows = _load_queries(Path(args.val)) if args.val else []
    corpus_rows = _load_corpus(Path(args.corpus))

    cfg = TrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        use_lora=not args.disable_lora,
        tracking_backend=args.tracking_backend,
        experiment_name=args.experiment_name,
    )
    manifest = train_biencoder(
        train_rows,
        val_rows,
        corpus_rows,
        Path(args.out_dir),
        cfg,
        corpus_hash=sha256_file(Path(args.corpus)),
    )
    print(json.dumps({"model_dir": str(Path(args.out_dir).resolve()), **manifest}, indent=2))


def _load_eval_rows(path: Path, split: str | None) -> list[QueryRecord]:
    rows = _load_queries(path)
    if split:
        return [r for r in rows if r.split == split]
    return rows


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    eval_rows = _load_eval_rows(Path(args.eval_set), args.split)
    corpus_rows = _load_corpus(Path(args.corpus))

    base_metrics = evaluate(
        eval_rows,
        corpus_rows,
        model_path_or_name=args.base_model,
        out_path=out_dir / "baseline_metrics.json",
    )
    tuned_metrics = evaluate(
        eval_rows,
        corpus_rows,
        model_path_or_name=args.tuned_model,
        out_path=out_dir / "tuned_metrics.json",
    )

    delta = {
        "recall@5": tuned_metrics["recall@5"] - base_metrics["recall@5"],
        "mrr@10": tuned_metrics["mrr@10"] - base_metrics["mrr@10"],
        "ndcg@10": tuned_metrics["ndcg@10"] - base_metrics["ndcg@10"],
    }

    summary = {"baseline": base_metrics, "tuned": tuned_metrics, "delta": delta}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_build_ann_index(args: argparse.Namespace) -> None:
    corpus_rows = _load_corpus(Path(args.corpus))
    config = build_index(corpus_rows, args.model, Path(args.out_dir))
    print(json.dumps(config, indent=2))


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="graphft", description="GraphQL finetuning pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest-schema")
    p_ingest.add_argument("--input", required=True)
    p_ingest.add_argument("--out-dir", required=True)
    p_ingest.set_defaults(func=cmd_ingest_schema)

    p_corpus = sub.add_parser("build-corpus")
    p_corpus.add_argument("--normalized", required=True)
    p_corpus.add_argument("--version", type=int, default=1)
    p_corpus.add_argument("--out-dir", required=True)
    p_corpus.set_defaults(func=cmd_build_corpus)

    p_gen = sub.add_parser("generate-synthetic")
    p_gen.add_argument("--corpus", required=True)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--max-expansions", type=int, default=6)
    p_gen.add_argument("--train-ratio", type=float, default=0.8)
    p_gen.add_argument("--val-ratio", type=float, default=0.1)
    p_gen.add_argument("--out-dir", required=True)
    p_gen.set_defaults(func=cmd_generate_synthetic)

    p_hn = sub.add_parser("mine-hard-negatives")
    p_hn.add_argument("--corpus", required=True)
    p_hn.add_argument("--queries", required=True)
    p_hn.add_argument("--hard-k", type=int, default=5)
    p_hn.add_argument("--medium-k", type=int, default=5)
    p_hn.add_argument("--easy-k", type=int, default=5)
    p_hn.add_argument("--out-dir", required=True)
    p_hn.set_defaults(func=cmd_mine_hard_negatives)

    p_train = sub.add_parser("train-embedder")
    p_train.add_argument("--train", required=True)
    p_train.add_argument("--val")
    p_train.add_argument("--corpus", required=True)
    p_train.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--learning-rate", type=float, default=2e-5)
    p_train.add_argument("--max-seq-length", type=int, default=512)
    p_train.add_argument("--disable-lora", action="store_true")
    p_train.add_argument("--tracking-backend", choices=["none", "wandb", "mlflow"], default="none")
    p_train.add_argument("--experiment-name", default="graphql-embedder-finetune")
    p_train.add_argument("--out-dir", required=True)
    p_train.set_defaults(func=cmd_train_embedder)

    p_eval = sub.add_parser("eval-retrieval")
    p_eval.add_argument("--eval-set", required=True)
    p_eval.add_argument("--split", choices=["train", "val", "test"])
    p_eval.add_argument("--corpus", required=True)
    p_eval.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
    p_eval.add_argument("--tuned-model", required=True)
    p_eval.add_argument("--out-dir", required=True)
    p_eval.set_defaults(func=cmd_eval_retrieval)

    p_idx = sub.add_parser("build-ann-index")
    p_idx.add_argument("--corpus", required=True)
    p_idx.add_argument("--model", required=True)
    p_idx.add_argument("--out-dir", required=True)
    p_idx.set_defaults(func=cmd_build_ann_index)

    return p


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
