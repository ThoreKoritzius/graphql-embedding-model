# GraphQL Finetuning Pipeline

Synthetic-schema-first pipeline for training an embedding model to retrieve relevant GraphQL types for user questions where users do **not** know schema/type names.

## Core approach

- Generate many synthetic schema worlds (domains like hotels/shopping/cars/fintech/etc.)
- Generate schema-oblivious user questions with multi-relevant type labels
- Train retrieval model with structural multi-positive supervision (`typename`, `Type->field`, `type ... { ... }`)
- Evaluate on held-out worlds with realism/adversarial/compositional benchmarks
- Track metrics and plots locally and optionally in W&B

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,wandb]'
```

## Environment

```bash
OPENAI_API_KEY=...
WANDB_API_KEY=...
```

Load vars:

```bash
source .venv/bin/activate
set -a; source .env; set +a
```

## Quick workflow

```bash
make ingest-schema
make generate-openai-seed
make build-dataset
make backfill-structural-corpus
make train-embedder
make eval-retrieval
make run-benchmark
make plot-metrics
make build-ann-index
```

## Production one-command flow

Use a single versioned run with hard checks (non-empty train/val/test):

```bash
make prod-run VERSION=11 CONFIG=examples/pipeline_config.prod.yaml TRAIN_SIZE=10000 EPOCHS=3
```

Outputs:
- model: `artifacts/models/qwen3-v11-sdl`
- eval: `artifacts/eval/v11`
- index: `artifacts/index/v11`

## Main commands

### 1) Generate synthetic schema worlds + seed queries

```bash
graphft generate-openai-seed \
  --out-dir artifacts \
  --version 1 \
  --config examples/pipeline_config.yaml \
  --request-batch-size 20 \
  --max-concurrency 4
```

Outputs:
- `artifacts/worlds/v1/world_<id>/schema.graphql`
- `artifacts/worlds/v1/world_<id>/type_catalog.json`
- `artifacts/worlds/v1/manifest.json`
- `artifacts/corpus/types_worlds_v1.jsonl`
- `artifacts/openai/seed_pairs_v1.jsonl`
 - live checkpoints during generation:
   - `artifacts/openai/seed_pairs_v1.partial.jsonl`
   - `artifacts/openai/raw_seed_responses.partial.jsonl`

Notes:
- `--request-batch-size` controls examples requested per API call.
- `--max-concurrency` controls parallel API calls per world.

### 2) Build train/val/test dataset from world-seeded queries

```bash
graphft build-dataset \
  --corpus artifacts/corpus/types_worlds_v1.jsonl \
  --openai-seed artifacts/openai/seed_pairs_v1.jsonl \
  --schema-hash-file artifacts/metadata/schema_hash.json \
  --out-dir artifacts \
  --version 1 \
  --config examples/pipeline_config.yaml
```

Outputs:
- `artifacts/datasets/v1/{train,val,test}.jsonl`
- `artifacts/datasets/v1/corpus.jsonl`
- `artifacts/datasets/v1/benchmarks/{realism_eval,adversarial_eval,compositional_eval}.jsonl`
- `artifacts/datasets/v1/manifest.json`

### 3) Train embedder (epoch benchmark logging)

If your corpus was generated before structural views existed, backfill first:

```bash
graphft backfill-structural-corpus \
  --corpus artifacts/datasets/v1/corpus.jsonl \
  --out artifacts/datasets/v1/corpus_structural.jsonl \
  --worlds-dir artifacts/worlds/v1 \
  --primary-retrieval-view sdl
```

```bash
graphft train-embedder \
  --train artifacts/datasets/v1/train.jsonl \
  --val artifacts/datasets/v1/val.jsonl \
  --corpus artifacts/datasets/v1/corpus_structural.jsonl \
  --model Qwen/Qwen3-Embedding-0.6B \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --disable-lora \
  --positive-views typename,field_paths,sdl \
  --primary-retrieval-view sdl \
  --tracking-backend wandb \
  --eval-every-epoch \
  --benchmark-dir artifacts/datasets/v1/benchmarks \
  --run-name qwen3-embedder-v1 \
  --out-dir artifacts/models/qwen3-embedding-0.6b-ft
```

### 4) Baseline vs tuned test-set evaluation

```bash
graphft eval-retrieval \
  --eval-set artifacts/datasets/v1/test.jsonl \
  --corpus artifacts/datasets/v1/corpus_structural.jsonl \
  --base-model Qwen/Qwen3-Embedding-0.6B \
  --tuned-model artifacts/models/qwen3-embedding-0.6b-ft \
  --retrieval-view sdl \
  --out-dir artifacts/eval
```

### 5) Benchmarks + plots

```bash
graphft run-benchmark \
  --benchmark-dir artifacts/datasets/v1/benchmarks \
  --corpus artifacts/datasets/v1/corpus_structural.jsonl \
  --model artifacts/models/qwen3-embedding-0.6b-ft \
  --retrieval-view sdl \
  --tracking-backend wandb \
  --out-dir artifacts/eval/benchmarks

graphft plot-metrics \
  --epoch-metrics artifacts/models/qwen3-embedding-0.6b-ft/metrics/epoch_metrics.jsonl \
  --benchmark-summary artifacts/eval/benchmarks/benchmarks_summary.json \
  --tracking-backend wandb \
  --out-dir artifacts/eval/plots
```

### 6) Build ANN index

```bash
graphft build-ann-index \
  --corpus artifacts/datasets/v1/corpus_structural.jsonl \
  --model artifacts/models/qwen3-embedding-0.6b-ft \
  --retrieval-view sdl \
  --out-dir artifacts/index
```

## Label format in dataset rows

Each query row can include:
- `world_id`, `domain`, `query`
- `primary_type_id`
- `relevant_type_ids` (2-5)
- `relation_pair` (`primary`, `bridge`)
- `difficulty`, `noise_tags`, `adversarial_tags`, `negative_type_ids`

## Structural corpus views

Each corpus row can include:
- `type_name_text` (`TypeName`)
- `field_paths_text` (`TypeName->field:type`)
- `sdl_text` (`type TypeName { ... }`)
- `retrieval_text` (selected primary retrieval view; default `sdl`)

## Metrics

- Core: `recall@1/5/10`, `mrr@10`, `ndcg@10`
- Set/pair: `set_recall_any@5`, `set_recall_all@10`, `coverage@10`, `pair_recall@10`
- Transfer: `seen_vs_unseen_recall@5_gap`

## Offline generation/testing

Use mocked OpenAI responses:

```bash
graphft generate-openai-seed --out-dir artifacts --version 1 --mock-responses-path /path/mock.jsonl
```

## Notes

- `eval-retrieval` and `build-ann-index` fail fast for invalid local model paths.
- If plotting fails due cache permissions in restricted envs, try:
  ```bash
  MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp graphft plot-metrics ...
  ```
