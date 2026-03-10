.PHONY: install test lint ingest-schema build-corpus generate-synthetic mine-hard-negatives train-embedder eval-retrieval build-ann-index

PYTHON ?= python3
GRAPHFT ?= graphft

install:
	$(PYTHON) -m pip install -e .[dev]

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

mine-hard-negatives:
	$(GRAPHFT) mine-hard-negatives --corpus artifacts/corpus/types_v1.jsonl --queries artifacts/splits/train.jsonl --out-dir artifacts

train-embedder:
	$(GRAPHFT) train-embedder --train artifacts/splits/train.jsonl --val artifacts/splits/val.jsonl --corpus artifacts/corpus/types_v1.jsonl --disable-lora --out-dir artifacts/models/qwen3-embedding-0.6b-ft

eval-retrieval:
	$(GRAPHFT) eval-retrieval --eval-set artifacts/splits/test.jsonl --corpus artifacts/corpus/types_v1.jsonl --base-model Qwen/Qwen3-Embedding-0.6B --tuned-model artifacts/models/qwen3-embedding-0.6b-ft --out-dir artifacts/eval

build-ann-index:
	$(GRAPHFT) build-ann-index --corpus artifacts/corpus/types_v1.jsonl --model artifacts/models/qwen3-embedding-0.6b-ft --out-dir artifacts/index
