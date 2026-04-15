from pathlib import Path

import pytest

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.models import QueryRecord
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.training.train_embedder import build_pair_dataset, compute_warmup_steps
from graphql_finetuning_pipeline.utils.embeddings import encode_with_resolution


def _build_rows():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    target = corpus[0].coordinate
    train_rows = [QueryRecord(query_id="q1", query="what field stores the identifier", positive_coordinate=target, split="train", source="test", family_id="f1", quality_score=1.0, owner_type=corpus[0].owner_type, field_name=corpus[0].field_name, relevant_coordinates=[target], negative_coordinates=[c.coordinate for c in corpus[1:3]])]
    return corpus, train_rows


def test_build_pair_dataset_contract():
    corpus, train_rows = _build_rows()
    coord_to_docs = {c.coordinate: [c.retrieval_text] for c in corpus}
    primary_docs = {c.coordinate: c.retrieval_text for c in corpus}
    ds = build_pair_dataset(train_rows, coord_to_docs, primary_docs)
    assert len(ds) > 0
    assert set(ds.column_names) == {"anchor", "positive", "negative"}


def test_compute_warmup_steps_is_positive_for_tiny_dataset():
    assert compute_warmup_steps(train_size=1, batch_size=16, epochs=1) >= 1


def test_local_model_resolution_fails_fast_for_missing_path():
    with pytest.raises(FileNotFoundError, match="Local model path does not exist"):
        encode_with_resolution("artifacts/models/does-not-exist", ["hello"])
