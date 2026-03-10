# GraphQL Finetuning Pipeline

Reproducible pipeline to finetune an embedding model (default: `Qwen/Qwen3-Embedding-0.6B`) for GraphQL type retrieval.

## Pipeline stages

- `ingest-schema`: parse GraphQL SDL/introspection and normalize type records
- `build-corpus`: generate deterministic retrieval docs (`short`, `full`, `keywords`) and versioned corpus
- `generate-synthetic`: bootstrap + expand synthetic query pairs with quality filters
  - Also writes benchmark sets: `synthetic_holdout`, `realism_seed`, `adversarial_ambiguity`
- `mine-hard-negatives`: attach easy/medium/hard negatives per query
- `train-embedder`: finetune bi-encoder with in-batch contrastive objective and optional LoRA
  - Optional experiment tracking: `--tracking-backend wandb|mlflow`
- `eval-retrieval`: compute Recall@K, MRR@10, nDCG@10 and baseline comparisons
- `build-ann-index`: build FAISS (if available) or sklearn cosine index

## Quickstart

```bash
python -m pip install -e .[dev]
make ingest-schema
make build-corpus
make generate-synthetic
make mine-hard-negatives
make train-embedder
make eval-retrieval
make build-ann-index
```

All outputs are stored under `artifacts/`.

Notes:
- `train-embedder --out-dir <path>` writes a loadable SentenceTransformer model directly at `<path>`.
- `eval-retrieval --tuned-model <path>` and `build-ann-index --model <path>` fail fast for missing/invalid local model paths.
- `make train-embedder` uses `--disable-lora` for stable checkpoint reload behavior in eval/index.
- LoRA remains available for manual runs; if adapter merge/load warnings appear, rerun with `--disable-lora`.

## Artifact layout

- `artifacts/normalized_types.jsonl`
- `artifacts/metadata/schema_hash.json`
- `artifacts/corpus/types_v1.jsonl`
- `artifacts/splits/{train,val,test}.jsonl`
- `artifacts/benchmarks/{synthetic_holdout,realism_seed,adversarial_ambiguity}.jsonl`
- `artifacts/models/.../training_manifest.json`
- `artifacts/eval/{baseline_metrics,tuned_metrics,summary}.json`
- `artifacts/index/{retrieval_config,ids}.json` + index binary
