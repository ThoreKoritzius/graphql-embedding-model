.PHONY: install test lint ingest-schema build-corpus backfill-structural-corpus generate-openai-seed build-dataset generate-synthetic mine-hard-negatives train-embedder eval-retrieval run-benchmark plot-metrics build-ann-index prod-check prod-run

PYTHON ?= python3
GRAPHFT ?= graphft
VERSION ?= 11
CONFIG ?= examples/pipeline_config.prod.yaml
TRAIN_SIZE ?= 10000
EPOCHS ?= 3
MODEL_OUT ?= artifacts/models/qwen3-v$(VERSION)-field-semantic
EVAL_OUT ?= artifacts/eval/v$(VERSION)
INDEX_OUT ?= artifacts/index/v$(VERSION)

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

backfill-structural-corpus:
	$(GRAPHFT) backfill-structural-corpus --corpus artifacts/datasets/v1/corpus.jsonl --out artifacts/datasets/v1/corpus_structural.jsonl --worlds-dir artifacts/worlds/v1 --primary-retrieval-view semantic

generate-synthetic:
	$(GRAPHFT) generate-synthetic --corpus artifacts/corpus/types_v1.jsonl --out-dir artifacts

generate-openai-seed:
	$(GRAPHFT) generate-openai-seed --out-dir artifacts --version 1 --config examples/pipeline_config.yaml

build-dataset:
	$(GRAPHFT) build-dataset --corpus artifacts/corpus/types_worlds_v1.jsonl --openai-seed artifacts/openai/seed_pairs_v1.jsonl --schema-hash-file artifacts/metadata/schema_hash.json --out-dir artifacts --version 1 --config examples/pipeline_config.yaml

mine-hard-negatives:
	$(GRAPHFT) mine-hard-negatives --corpus artifacts/corpus/types_v1.jsonl --queries artifacts/splits/train.jsonl --out-dir artifacts

train-embedder:
	$(GRAPHFT) train-embedder --train artifacts/datasets/v1/train.jsonl --val artifacts/datasets/v1/val.jsonl --corpus artifacts/datasets/v1/corpus.jsonl --disable-lora --positive-views coordinate,signature,semantic,sdl --primary-retrieval-view semantic --eval-every-epoch --benchmark-dir artifacts/datasets/v1/benchmarks --tracking-backend wandb --run-name qwen3-embedder-v1 --out-dir artifacts/models/qwen3-field-embedding-0.6b-ft

eval-retrieval:
	$(GRAPHFT) eval-retrieval --eval-set artifacts/datasets/v1/test.jsonl --corpus artifacts/datasets/v1/corpus.jsonl --base-model Qwen/Qwen3-Embedding-0.6B --tuned-model artifacts/models/qwen3-field-embedding-0.6b-ft --retrieval-view semantic --out-dir artifacts/eval

run-benchmark:
	$(GRAPHFT) run-benchmark --benchmark-dir artifacts/datasets/v1/benchmarks --corpus artifacts/datasets/v1/corpus.jsonl --model artifacts/models/qwen3-field-embedding-0.6b-ft --retrieval-view semantic --tracking-backend wandb --out-dir artifacts/eval/benchmarks

plot-metrics:
	$(GRAPHFT) plot-metrics --epoch-metrics artifacts/models/qwen3-field-embedding-0.6b-ft/metrics/epoch_metrics.jsonl --benchmark-summary artifacts/eval/benchmarks/benchmarks_summary.json --tracking-backend wandb --out-dir artifacts/eval/plots

build-ann-index:
	$(GRAPHFT) build-ann-index --corpus artifacts/datasets/v1/corpus.jsonl --model artifacts/models/qwen3-field-embedding-0.6b-ft --retrieval-view semantic --out-dir artifacts/index

prod-check:
	@test -f artifacts/datasets/v$(VERSION)/train.jsonl
	@test -f artifacts/datasets/v$(VERSION)/val.jsonl
	@test -f artifacts/datasets/v$(VERSION)/test.jsonl
	@test $$(wc -l < artifacts/datasets/v$(VERSION)/train.jsonl) -gt 0
	@test $$(wc -l < artifacts/datasets/v$(VERSION)/val.jsonl) -gt 0
	@test $$(wc -l < artifacts/datasets/v$(VERSION)/test.jsonl) -gt 0

prod-run:
	$(GRAPHFT) generate-openai-seed --out-dir artifacts --version $(VERSION) --config $(CONFIG) --world-count 60 --items-per-world 200 --request-batch-size 20 --max-concurrency 2
	$(GRAPHFT) build-dataset --corpus artifacts/corpus/types_worlds_v$(VERSION).jsonl --openai-seed artifacts/openai/seed_pairs_v1.jsonl --out-dir artifacts --version $(VERSION) --config $(CONFIG) --target-train-min $(TRAIN_SIZE) --target-train-max $(TRAIN_SIZE)
	$(MAKE) prod-check VERSION=$(VERSION)
	$(GRAPHFT) backfill-structural-corpus --corpus artifacts/datasets/v$(VERSION)/corpus.jsonl --out artifacts/datasets/v$(VERSION)/corpus_structural.jsonl --worlds-dir artifacts/worlds/v$(VERSION) --primary-retrieval-view semantic
	$(GRAPHFT) train-embedder --train artifacts/datasets/v$(VERSION)/train.jsonl --val artifacts/datasets/v$(VERSION)/val.jsonl --corpus artifacts/datasets/v$(VERSION)/corpus_structural.jsonl --model Qwen/Qwen3-Embedding-0.6B --positive-views semantic --primary-retrieval-view semantic --epochs $(EPOCHS) --disable-lora --tracking-backend wandb --eval-every-epoch --benchmark-dir artifacts/datasets/v$(VERSION)/benchmarks --out-dir $(MODEL_OUT)
	$(GRAPHFT) eval-retrieval --eval-set artifacts/datasets/v$(VERSION)/test.jsonl --corpus artifacts/datasets/v$(VERSION)/corpus_structural.jsonl --base-model Qwen/Qwen3-Embedding-0.6B --tuned-model $(MODEL_OUT) --retrieval-view semantic --out-dir $(EVAL_OUT)
	$(GRAPHFT) build-ann-index --corpus artifacts/datasets/v$(VERSION)/corpus_structural.jsonl --model $(MODEL_OUT) --retrieval-view semantic --out-dir $(INDEX_OUT)
