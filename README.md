# GraphQL Finetuning Pipeline

Pipeline for finetuning embedding models (default: `Qwen/Qwen3-Embedding-0.6B`) to retrieve GraphQL types from user queries.

## What it supports

- Schema ingestion and corpus construction from SDL/introspection
- OpenAI-seeded query generation (`gpt-4o-mini` default)
- Deterministic local augmentation + quality gates
- Versioned dataset building (`artifacts/datasets/v{n}`)
- Bi-encoder finetuning with optional epoch-end benchmark evaluation
- Baseline-vs-tuned retrieval eval
- Multi-suite benchmark evaluation with slice metrics
- Local plot generation + optional W&B logging
- ANN index build for deployment

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,wandb]'
```

## Environment

Create `.env` with at least:

```bash
OPENAI_API_KEY=...
WANDB_API_KEY=...
```

Load env vars:

```bash
source .venv/bin/activate
set -a; source .env; set +a
```

## Fast path (Make targets)

```bash
make ingest-schema
make build-corpus
make generate-openai-seed
make build-dataset
make train-embedder
make eval-retrieval
make run-benchmark
make plot-metrics
make build-ann-index
```

## Full CLI examples

### 1) Ingest and build corpus

```bash
graphft ingest-schema --input examples/schema.graphql --out-dir artifacts
graphft build-corpus --normalized artifacts/normalized_types.jsonl --out-dir artifacts
```

### 2) OpenAI seed generation

```bash
graphft generate-openai-seed \
  --corpus artifacts/corpus/types_v1.jsonl \
  --out-dir artifacts \
  --config examples/pipeline_config.yaml
```

Outputs:
- `artifacts/openai/raw_seed_responses.jsonl`
- `artifacts/openai/seed_pairs_v1.jsonl`

### 3) Build versioned dataset

```bash
graphft build-dataset \
  --corpus artifacts/corpus/types_v1.jsonl \
  --openai-seed artifacts/openai/seed_pairs_v1.jsonl \
  --schema-hash-file artifacts/metadata/schema_hash.json \
  --out-dir artifacts \
  --version 1 \
  --config examples/pipeline_config.yaml
```

Outputs:
- `artifacts/datasets/v1/{train,val,test}.jsonl`
- `artifacts/datasets/v1/benchmarks/{synthetic_holdout,realism_eval,adversarial_eval}.jsonl`
- `artifacts/datasets/v1/manifest.json`

### 4) Train with W&B and epoch benchmark logging

```bash
graphft train-embedder \
  --train artifacts/datasets/v1/train.jsonl \
  --val artifacts/datasets/v1/val.jsonl \
  --corpus artifacts/corpus/types_v1.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --disable-lora \
  --tracking-backend wandb \
  --eval-every-epoch \
  --benchmark-dir artifacts/datasets/v1/benchmarks \
  --run-name qwen3-embedder-v1 \
  --out-dir artifacts/models/qwen3-embedding-0.6b-ft
```

Epoch metrics output:
- `artifacts/models/qwen3-embedding-0.6b-ft/metrics/epoch_metrics.jsonl`

### 5) Baseline vs tuned eval

```bash
graphft eval-retrieval \
  --eval-set artifacts/datasets/v1/test.jsonl \
  --corpus artifacts/corpus/types_v1.jsonl \
  --base-model Qwen/Qwen3-Embedding-0.6B \
  --tuned-model artifacts/models/qwen3-embedding-0.6b-ft \
  --out-dir artifacts/eval
```

### 6) Benchmark suites and plots

```bash
graphft run-benchmark \
  --benchmark-dir artifacts/datasets/v1/benchmarks \
  --corpus artifacts/corpus/types_v1.jsonl \
  --model artifacts/models/qwen3-embedding-0.6b-ft \
  --tracking-backend wandb \
  --out-dir artifacts/eval/benchmarks

graphft plot-metrics \
  --epoch-metrics artifacts/models/qwen3-embedding-0.6b-ft/metrics/epoch_metrics.jsonl \
  --benchmark-summary artifacts/eval/benchmarks/benchmarks_summary.json \
  --tracking-backend wandb \
  --out-dir artifacts/eval/plots
```

### 7) Build ANN index

```bash
graphft build-ann-index \
  --corpus artifacts/corpus/types_v1.jsonl \
  --model artifacts/models/qwen3-embedding-0.6b-ft \
  --out-dir artifacts/index
```

## Create ~100 OpenAI-seeded training entries (example)

```bash
graphft generate-openai-seed \
  --corpus artifacts/corpus/types_v1.jsonl \
  --out-dir artifacts \
  --items-per-type 10

graphft build-dataset \
  --corpus artifacts/corpus/types_v1.jsonl \
  --openai-seed artifacts/openai/seed_pairs_v1.jsonl \
  --schema-hash-file artifacts/metadata/schema_hash.json \
  --out-dir artifacts \
  --version 3 \
  --target-train-min 100 \
  --target-train-max 100
```

Check counts:

```bash
wc -l artifacts/datasets/v3/train.jsonl artifacts/datasets/v3/val.jsonl artifacts/datasets/v3/test.jsonl
```

## Local synthetic mode (no OpenAI)

```bash
graphft generate-synthetic \
  --seed-source local \
  --corpus artifacts/corpus/types_v1.jsonl \
  --out-dir artifacts
```

## Troubleshooting

- `FileNotFoundError: Local model path does not exist ...`
  - Use an existing trained model dir under `artifacts/models/`.
- Plotting crashes in restricted environments (font cache issues)
  - Try:
    ```bash
    MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp graphft plot-metrics ...
    ```
- If OpenAI generation appears idle
  - `generate-openai-seed` shows a `tqdm` progress bar per type (`OpenAI seed generation ...`).

## Notes

- `train-embedder --out-dir <path>` writes a loadable SentenceTransformer model at `<path>`.
- `eval-retrieval --tuned-model <path>` and `build-ann-index --model <path>` fail fast for missing/invalid local paths.
- `--disable-lora` is recommended for stable reload behavior in eval/index.
- Offline testing of OpenAI generation is supported via `--mock-responses-path <jsonl>`.
