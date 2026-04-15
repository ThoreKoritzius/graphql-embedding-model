import pytest

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.eval.retrieval_eval import evaluate


def test_eval_fails_fast_when_requested_view_missing():
    corpus = [CorpusRecord(doc_id="T.id", coordinate="T.id", owner_type="T", field_name="id", return_type="ID", short_text="x", full_text="x", keywords_text="x", coordinate_text="T.id", field_signature_text="", field_semantic_text="", sdl_snippet_text="", retrieval_text="", metadata={})]
    rows = [QueryRecord(query_id="q1", query="find something", positive_coordinate="T.id", split="test", source="test", family_id="f1", quality_score=1.0, owner_type="T", field_name="id")]
    with pytest.raises(ValueError, match="backfill-structural-corpus"):
        evaluate(rows, corpus, "Qwen/Qwen3-Embedding-0.6B", retrieval_view="sdl")


def test_eval_empty_rows_returns_zero_metrics():
    corpus = [CorpusRecord(doc_id="T.id", coordinate="T.id", owner_type="T", field_name="id", return_type="ID", short_text="x", full_text="x", keywords_text="x", coordinate_text="T.id", field_signature_text="T.id: ID", field_semantic_text="GraphQL field T.id", sdl_snippet_text="type T {\n  id: ID\n}", retrieval_text="GraphQL field T.id", metadata={})]
    out = evaluate([], corpus, "Qwen/Qwen3-Embedding-0.6B", retrieval_view="semantic")
    assert out["count"] == 0
    assert out["exact_match@1"] == 0.0
