.PHONY: install test lint ingest-schema build-corpus generate-openai-seed build-dataset generate-synthetic mine-hard-negatives train-embedder eval-retrieval run-benchmark plot-metrics build-ann-index

PYTHON ?= python3
GRAPHFT ?= graphft

install:
	$(PYTHON) -m pip install -e .[dev,wandb]

test:
	pytest

lint:
	ruff check graphql_finetuning_pipeline tests

ingest-schema:
	$(GRAPHFT) ingest-schema --input examples/schema.graphql --out-dir artifacts

build-corpus:
	$(GRAPHFT) build-corpus --normalized artifacts/normalized_types.jsonl --out-dir artifacts

generate-synthetic:
	$(GRAPHFT) generate-synthetic --corpus artifacts/corpus/types_v1.jsonl --out-dir artifacts

generate-openai-seed:
	$(GRAPHFT) generate-openai-seed --out-dir artifacts --version 1 --config examples/pipeline_config.yaml

build-dataset:
	$(GRAPHFT) build-dataset --corpus artifacts/corpus/types_worlds_v1.jsonl --openai-seed artifacts/openai/seed_pairs_v1.jsonl --schema-hash-file artifacts/metadata/schema_hash.json --out-dir artifacts --version 1 --config examples/pipeline_config.yaml

mine-hard-negatives:
	$(GRAPHFT) mine-hard-negatives --corpus artifacts/corpus/types_v1.jsonl --queries artifacts/splits/train.jsonl --out-dir artifacts

train-embedder:
	$(GRAPHFT) train-embedder --train artifacts/datasets/v1/train.jsonl --val artifacts/datasets/v1/val.jsonl --corpus artifacts/datasets/v1/corpus.jsonl --disable-lora --eval-every-epoch --benchmark-dir artifacts/datasets/v1/benchmarks --tracking-backend wandb --run-name qwen3-embedder-v1 --out-dir artifacts/models/qwen3-embedding-0.6b-ft

eval-retrieval:
	$(GRAPHFT) eval-retrieval --eval-set artifacts/datasets/v1/test.jsonl --corpus artifacts/datasets/v1/corpus.jsonl --base-model Qwen/Qwen3-Embedding-0.6B --tuned-model artifacts/models/qwen3-embedding-0.6b-ft --out-dir artifacts/eval

run-benchmark:
	$(GRAPHFT) run-benchmark --benchmark-dir artifacts/datasets/v1/benchmarks --corpus artifacts/datasets/v1/corpus.jsonl --model artifacts/models/qwen3-embedding-0.6b-ft --tracking-backend wandb --out-dir artifacts/eval/benchmarks

plot-metrics:
	$(GRAPHFT) plot-metrics --epoch-metrics artifacts/models/qwen3-embedding-0.6b-ft/metrics/epoch_metrics.jsonl --benchmark-summary artifacts/eval/benchmarks/benchmarks_summary.json --tracking-backend wandb --out-dir artifacts/eval/plots

build-ann-index:
	$(GRAPHFT) build-ann-index --corpus artifacts/datasets/v1/corpus.jsonl --model artifacts/models/qwen3-embedding-0.6b-ft --out-dir artifacts/index
