from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.corpus_backfill import backfill_structural_views, load_world_type_lookup
from graphql_finetuning_pipeline.data.dataset_builder import DatasetBuildConfig, build_dataset
from graphql_finetuning_pipeline.data.models import CorpusRecord, GraphQLTypeRecord, QueryRecord
from graphql_finetuning_pipeline.data.openai_seed import OpenAISeedConfig, generate_openai_seed
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.structural_views import corpus_structural_hash, parse_positive_views
from graphql_finetuning_pipeline.data.synthetic import (
    bootstrap_queries,
    build_benchmark_sets,
    expand_queries,
    quality_filter,
    split_queries,
)
from graphql_finetuning_pipeline.eval.benchmark import run_benchmarks
from graphql_finetuning_pipeline.eval.plots import plot_benchmark_comparison, plot_epoch_metrics
from graphql_finetuning_pipeline.eval.retrieval_eval import evaluate
from graphql_finetuning_pipeline.retrieval.hard_negatives import mine_hard_negatives
from graphql_finetuning_pipeline.retrieval.index import build_index
from graphql_finetuning_pipeline.utils.io import ensure_dir, read_json, read_jsonl, sha256_file, write_json, write_jsonl


def _load_types(path: Path) -> list[GraphQLTypeRecord]:
    return [GraphQLTypeRecord.model_validate(x) for x in read_jsonl(path)]


def _load_corpus(path: Path) -> list[CorpusRecord]:
    return [CorpusRecord.model_validate(x) for x in read_jsonl(path)]


def _load_queries(path: Path) -> list[QueryRecord]:
    return [QueryRecord.model_validate(x) for x in read_jsonl(path)]


def _load_yaml(path: Path | None) -> dict:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config must be a mapping at the top level")
    return payload


def _load_benchmark_sets(benchmark_dir: Path | None) -> dict[str, list[QueryRecord]]:
    if not benchmark_dir:
        return {}
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")
    out: dict[str, list[QueryRecord]] = {}
    for f in sorted(benchmark_dir.glob("*.jsonl")):
        out[f.stem] = _load_queries(f)
    return out


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


def cmd_backfill_structural_corpus(args: argparse.Namespace) -> None:
    in_path = Path(args.corpus)
    out_path = Path(args.out)
    corpus_rows = _load_corpus(in_path)

    world_lookup = None
    if args.worlds_dir:
        world_lookup = load_world_type_lookup(Path(args.worlds_dir))

    out_rows = backfill_structural_views(
        corpus_rows,
        primary_retrieval_view=args.primary_retrieval_view,
        world_type_lookup=world_lookup,
    )
    ensure_dir(out_path.parent)
    write_jsonl(out_path, [r.model_dump() for r in out_rows])
    print(
        json.dumps(
            {
                "input_rows": len(corpus_rows),
                "output_rows": len(out_rows),
                "out_path": str(out_path),
                "primary_retrieval_view": args.primary_retrieval_view,
                "corpus_view_version": 1,
                "corpus_structural_hash": corpus_structural_hash(out_rows),
            },
            indent=2,
        )
    )


def cmd_generate_openai_seed(args: argparse.Namespace) -> None:
    ycfg = _load_yaml(Path(args.config) if args.config else None)
    ocfg = ycfg.get("openai", {})
    wcfg = ycfg.get("worlds", {})

    cfg = OpenAISeedConfig(
        model=args.model or ocfg.get("model", "gpt-4o-mini"),
        temperature=args.temperature if args.temperature is not None else float(ocfg.get("temperature", 0.7)),
        retries=args.retries if args.retries is not None else int(ocfg.get("retries", 3)),
        items_per_world=args.items_per_world if args.items_per_world is not None else int(ocfg.get("items_per_world", 240)),
        request_batch_size=args.request_batch_size if args.request_batch_size is not None else int(ocfg.get("request_batch_size", 40)),
        max_concurrency=args.max_concurrency if args.max_concurrency is not None else int(ocfg.get("max_concurrency", 4)),
        worlds_version=args.version,
        world_count=args.world_count if args.world_count is not None else int(wcfg.get("world_count", 60)),
        min_types_per_world=args.min_types if args.min_types is not None else int(wcfg.get("min_types", 20)),
        max_types_per_world=args.max_types if args.max_types is not None else int(wcfg.get("max_types", 36)),
    )

    rows, raw, corpus_rows = generate_openai_seed(
        out_dir=Path(args.out_dir),
        cfg=cfg,
        seed=args.seed,
        api_key=args.api_key,
        mock_responses_path=Path(args.mock_responses_path) if args.mock_responses_path else None,
    )
    print(
        json.dumps(
            {
                "seed_pairs": len(rows),
                "raw_logs": len(raw),
                "worlds_version": args.version,
                "world_count": cfg.world_count,
                "request_batch_size": cfg.request_batch_size,
                "max_concurrency": cfg.max_concurrency,
                "corpus_rows": len(corpus_rows),
                "seed_pairs_path": str(Path(args.out_dir) / "openai" / "seed_pairs_v1.jsonl"),
                "raw_path": str(Path(args.out_dir) / "openai" / "raw_seed_responses.jsonl"),
                "world_manifest": str(Path(args.out_dir) / "worlds" / f"v{args.version}" / "manifest.json"),
                "generated_corpus": str(Path(args.out_dir) / "corpus" / f"types_worlds_v{args.version}.jsonl"),
            },
            indent=2,
        )
    )


def cmd_build_dataset(args: argparse.Namespace) -> None:
    corpus_path = Path(args.corpus) if args.corpus else Path(args.out_dir) / "corpus" / f"types_worlds_v{args.version}.jsonl"
    corpus = _load_corpus(corpus_path)
    seed_rows = _load_queries(Path(args.openai_seed))
    cfg_yaml = _load_yaml(Path(args.config) if args.config else None)
    dcfg = cfg_yaml.get("dataset", {})

    cfg = DatasetBuildConfig(
        version=args.version,
        seed=args.seed,
        target_train_min=args.target_train_min if args.target_train_min is not None else int(dcfg.get("target_train_min", 20000)),
        target_train_max=args.target_train_max if args.target_train_max is not None else int(dcfg.get("target_train_max", 50000)),
    )

    schema_hash = ""
    if args.schema_hash_file:
        payload = read_json(Path(args.schema_hash_file))
        schema_hash = payload.get("schema_hash", "")
    elif args.schema_hash:
        schema_hash = args.schema_hash

    generation_config = {
        "openai_model": args.openai_model,
        "seed_source": "openai",
        "config_file": args.config,
    }
    manifest = build_dataset(
        openai_seed_rows=seed_rows,
        corpus=corpus,
        out_dir=Path(args.out_dir),
        schema_hash=schema_hash,
        cfg=cfg,
        generation_config=generation_config,
    )
    print(json.dumps({**manifest, "corpus_path": str(corpus_path)}, indent=2))


def cmd_generate_synthetic(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir / "splits")
    ensure_dir(out_dir / "benchmarks")

    corpus = _load_corpus(Path(args.corpus))
    if args.seed_source == "openai":
        if not args.openai_seed:
            raise ValueError("--openai-seed is required when --seed-source openai")
        seed = _load_queries(Path(args.openai_seed))
    else:
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
                "seed_source": args.seed_source,
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
    benchmark_sets = _load_benchmark_sets(Path(args.benchmark_dir) if args.benchmark_dir else None)

    run_name = args.run_name or args.experiment_name
    cfg = TrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        use_lora=not args.disable_lora,
        tracking_backend=args.tracking_backend,
        experiment_name=run_name,
        eval_every_epoch=args.eval_every_epoch,
        positive_views=parse_positive_views(args.positive_views),
        primary_retrieval_view=args.primary_retrieval_view,
    )
    manifest = train_biencoder(
        train_rows,
        val_rows,
        corpus_rows,
        Path(args.out_dir),
        cfg,
        corpus_hash=sha256_file(Path(args.corpus)),
        benchmark_sets=benchmark_sets,
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
    if not eval_rows:
        raise ValueError(
            f"No evaluation rows found in {args.eval_set}"
            + (f" for split={args.split}" if args.split else "")
            + ". Use a non-empty eval file (e.g., val split) or rebuild dataset with held-out test rows."
        )
    corpus_rows = _load_corpus(Path(args.corpus))

    base_metrics = evaluate(
        eval_rows,
        corpus_rows,
        model_path_or_name=args.base_model,
        out_path=out_dir / "baseline_metrics.json",
        retrieval_view=args.retrieval_view,
    )
    tuned_metrics = evaluate(
        eval_rows,
        corpus_rows,
        model_path_or_name=args.tuned_model,
        out_path=out_dir / "tuned_metrics.json",
        retrieval_view=args.retrieval_view,
    )

    delta = {
        "exact_match@1": tuned_metrics["exact_match@1"] - base_metrics["exact_match@1"],
        "recall@5": tuned_metrics["recall@5"] - base_metrics["recall@5"],
        "mrr@10": tuned_metrics["mrr@10"] - base_metrics["mrr@10"],
        "ndcg@10": tuned_metrics["ndcg@10"] - base_metrics["ndcg@10"],
    }

    summary = {"baseline": base_metrics, "tuned": tuned_metrics, "delta": delta}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_run_benchmark(args: argparse.Namespace) -> None:
    corpus_rows = _load_corpus(Path(args.corpus))
    out_dir = Path(args.out_dir)
    summary = run_benchmarks(
        Path(args.benchmark_dir),
        corpus_rows,
        args.model,
        out_dir,
        retrieval_view=args.retrieval_view,
    )

    if args.tracking_backend == "wandb":
        try:
            import wandb

            table = wandb.Table(columns=["benchmark", "exact_match@1", "recall@5", "mrr@10", "ndcg@10", "same_owner_wrong_field_rate@1", "count"])
            for name, vals in summary.items():
                table.add_data(
                    name,
                    vals.get("exact_match@1", 0.0),
                    vals["recall@5"],
                    vals["mrr@10"],
                    vals["ndcg@10"],
                    vals.get("same_owner_wrong_field_rate@1", 0.0),
                    vals["count"],
                )
                wandb.log(
                    {
                        f"benchmark/{name}/exact_match@1": vals.get("exact_match@1", 0.0),
                        f"benchmark/{name}/recall@5": vals["recall@5"],
                        f"benchmark/{name}/mrr@10": vals["mrr@10"],
                        f"benchmark/{name}/ndcg@10": vals["ndcg@10"],
                        f"benchmark/{name}/same_owner_wrong_field_rate@1": vals.get("same_owner_wrong_field_rate@1", 0.0),
                    }
                )
            wandb.log({"benchmark/summary_table": table})
        except Exception:
            pass

    print(json.dumps(summary, indent=2))


def cmd_plot_metrics(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    output = {}
    if args.epoch_metrics:
        output.update(plot_epoch_metrics(Path(args.epoch_metrics), out_dir))
    if args.benchmark_summary:
        output.update(plot_benchmark_comparison(Path(args.benchmark_summary), out_dir))

    if args.tracking_backend == "wandb":
        try:
            import wandb

            for key, path in output.items():
                if path.endswith(".png"):
                    wandb.log({key: wandb.Image(path)})
        except Exception:
            pass

    print(json.dumps(output, indent=2))


def cmd_build_ann_index(args: argparse.Namespace) -> None:
    corpus_rows = _load_corpus(Path(args.corpus))
    config = build_index(corpus_rows, args.model, Path(args.out_dir), retrieval_view=args.retrieval_view)
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

    p_backfill = sub.add_parser("backfill-structural-corpus")
    p_backfill.add_argument("--corpus", required=True)
    p_backfill.add_argument("--out", required=True)
    p_backfill.add_argument("--worlds-dir")
    p_backfill.add_argument("--primary-retrieval-view", choices=["coordinate", "signature", "semantic", "sdl"], default="semantic")
    p_backfill.set_defaults(func=cmd_backfill_structural_corpus)

    p_seed = sub.add_parser("generate-openai-seed")
    p_seed.add_argument("--out-dir", required=True)
    p_seed.add_argument("--config")
    p_seed.add_argument("--version", type=int, default=1)
    p_seed.add_argument("--world-count", type=int)
    p_seed.add_argument("--min-types", type=int)
    p_seed.add_argument("--max-types", type=int)
    p_seed.add_argument("--model")
    p_seed.add_argument("--temperature", type=float)
    p_seed.add_argument("--retries", type=int)
    p_seed.add_argument("--items-per-world", type=int)
    p_seed.add_argument("--request-batch-size", type=int)
    p_seed.add_argument("--max-concurrency", type=int)
    p_seed.add_argument("--seed", type=int, default=42)
    p_seed.add_argument("--api-key")
    p_seed.add_argument("--mock-responses-path")
    p_seed.set_defaults(func=cmd_generate_openai_seed)

    p_ds = sub.add_parser("build-dataset")
    p_ds.add_argument("--corpus")
    p_ds.add_argument("--openai-seed", required=True)
    p_ds.add_argument("--out-dir", required=True)
    p_ds.add_argument("--config")
    p_ds.add_argument("--version", type=int, default=1)
    p_ds.add_argument("--seed", type=int, default=42)
    p_ds.add_argument("--target-train-min", type=int)
    p_ds.add_argument("--target-train-max", type=int)
    p_ds.add_argument("--schema-hash")
    p_ds.add_argument("--schema-hash-file")
    p_ds.add_argument("--openai-model", default="gpt-4o-mini")
    p_ds.set_defaults(func=cmd_build_dataset)

    p_gen = sub.add_parser("generate-synthetic")
    p_gen.add_argument("--corpus", required=True)
    p_gen.add_argument("--seed-source", choices=["local", "openai"], default="local")
    p_gen.add_argument("--openai-seed")
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
    p_train.add_argument("--run-name")
    p_train.add_argument("--eval-every-epoch", action="store_true")
    p_train.add_argument("--benchmark-dir")
    p_train.add_argument("--positive-views", default="coordinate,signature,semantic,sdl")
    p_train.add_argument("--primary-retrieval-view", choices=["coordinate", "signature", "semantic", "sdl"], default="semantic")
    p_train.add_argument("--out-dir", required=True)
    p_train.set_defaults(func=cmd_train_embedder)

    p_eval = sub.add_parser("eval-retrieval")
    p_eval.add_argument("--eval-set", required=True)
    p_eval.add_argument("--split", choices=["train", "val", "test"])
    p_eval.add_argument("--corpus", required=True)
    p_eval.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
    p_eval.add_argument("--tuned-model", required=True)
    p_eval.add_argument("--retrieval-view", choices=["coordinate", "signature", "semantic", "sdl"], default="semantic")
    p_eval.add_argument("--out-dir", required=True)
    p_eval.set_defaults(func=cmd_eval_retrieval)

    p_bench = sub.add_parser("run-benchmark")
    p_bench.add_argument("--benchmark-dir", required=True)
    p_bench.add_argument("--corpus", required=True)
    p_bench.add_argument("--model", required=True)
    p_bench.add_argument("--retrieval-view", choices=["coordinate", "signature", "semantic", "sdl"], default="semantic")
    p_bench.add_argument("--tracking-backend", choices=["none", "wandb"], default="none")
    p_bench.add_argument("--out-dir", required=True)
    p_bench.set_defaults(func=cmd_run_benchmark)

    p_plot = sub.add_parser("plot-metrics")
    p_plot.add_argument("--epoch-metrics")
    p_plot.add_argument("--benchmark-summary")
    p_plot.add_argument("--tracking-backend", choices=["none", "wandb"], default="none")
    p_plot.add_argument("--out-dir", required=True)
    p_plot.set_defaults(func=cmd_plot_metrics)

    p_idx = sub.add_parser("build-ann-index")
    p_idx.add_argument("--corpus", required=True)
    p_idx.add_argument("--model", required=True)
    p_idx.add_argument("--retrieval-view", choices=["coordinate", "signature", "semantic", "sdl"], default="semantic")
    p_idx.add_argument("--out-dir", required=True)
    p_idx.set_defaults(func=cmd_build_ann_index)

    return p


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
