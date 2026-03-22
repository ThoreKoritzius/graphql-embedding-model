import pytest

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.eval.retrieval_eval import evaluate


def test_eval_fails_fast_when_requested_view_missing():
    corpus = [
        CorpusRecord(
            type_id="T1",
            type_name="T1",
            short_text="x",
            full_text="x",
            keywords_text="x",
            type_name_text="T1",
            field_paths_text="",
            sdl_text="",
            retrieval_text="",
            metadata={},
        )
    ]
    rows = [
        QueryRecord(
            query_id="q1",
            query="find something",
            target_type_id="T1",
            split="test",
            source="test",
            family_id="f1",
            quality_score=1.0,
        )
    ]
    with pytest.raises(ValueError, match="backfill-structural-corpus"):
        evaluate(rows, corpus, "Qwen/Qwen3-Embedding-0.6B", retrieval_view="sdl")


def test_eval_empty_rows_returns_zero_metrics():
    corpus = [
        CorpusRecord(
            type_id="T1",
            type_name="T1",
            short_text="x",
            full_text="x",
            keywords_text="x",
            type_name_text="T1",
            field_paths_text="T1->id:ID!",
            sdl_text="type T1 {\n  id: ID!\n}",
            retrieval_text="type T1 {\n  id: ID!\n}",
            metadata={},
        )
    ]
    out = evaluate([], corpus, "Qwen/Qwen3-Embedding-0.6B", retrieval_view="sdl")
    assert out["count"] == 0
    assert out["recall@5"] == 0.0
