from pathlib import Path

import pytest

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.synthetic import bootstrap_queries, expand_queries, quality_filter, split_queries
from graphql_finetuning_pipeline.training.train_embedder import build_pair_dataset, compute_warmup_steps
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


def _build_rows():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    seed = bootstrap_queries(corpus, rng_seed=1)
    expanded = expand_queries(seed)
    filtered = quality_filter(expanded, type_names={c.type_name for c in corpus}, leakage_threshold=1.0)
    split_rows = split_queries(filtered, seed=2)
    train_rows = [r for r in split_rows if r.split == "train"]
    return corpus, train_rows


def test_build_pair_dataset_contract():
    corpus, train_rows = _build_rows()
    type_to_doc = {c.type_id: c.full_text for c in corpus}
    ds = build_pair_dataset(train_rows, type_to_doc)
    assert len(ds) > 0
    assert set(ds.column_names) == {"anchor", "positive"}


def test_compute_warmup_steps_is_positive_for_tiny_dataset():
    assert compute_warmup_steps(train_size=1, batch_size=16, epochs=1) >= 1


def test_local_model_resolution_fails_fast_for_missing_path():
    with pytest.raises(FileNotFoundError, match="Local model path does not exist"):
        encode_with_resolution("artifacts/models/does-not-exist", ["hello"])
