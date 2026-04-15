import pytest

from graphql_finetuning_pipeline.data.dataset_builder import DatasetBuildConfig, ambiguity_filter, build_benchmark_suites, build_dataset, leakage_filter, semantic_dedupe
from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord


def _corpus() -> list[CorpusRecord]:
    return [
        CorpusRecord(doc_id="User.email", coordinate="User.email", owner_type="User", field_name="email", return_type="String", short_text="", full_text="GraphQL field User.email", keywords_text="", coordinate_text="User.email", field_signature_text="User.email: String", field_semantic_text="GraphQL field User.email", sdl_snippet_text="type User {\n  email: String\n}", retrieval_text="GraphQL field User.email", metadata={"world_id": "w1", "domain": "support"}),
        CorpusRecord(doc_id="User.name", coordinate="User.name", owner_type="User", field_name="name", return_type="String", short_text="", full_text="GraphQL field User.name", keywords_text="", coordinate_text="User.name", field_signature_text="User.name: String", field_semantic_text="GraphQL field User.name", sdl_snippet_text="type User {\n  name: String\n}", retrieval_text="GraphQL field User.name", metadata={"world_id": "w1", "domain": "support"}),
        CorpusRecord(doc_id="Ticket.status", coordinate="Ticket.status", owner_type="Ticket", field_name="status", return_type="String", short_text="", full_text="GraphQL field Ticket.status", keywords_text="", coordinate_text="Ticket.status", field_signature_text="Ticket.status: String", field_semantic_text="GraphQL field Ticket.status", sdl_snippet_text="type Ticket {\n  status: String\n}", retrieval_text="GraphQL field Ticket.status", metadata={"world_id": "w2", "domain": "support"}),
    ]


def _row(query_id: str, query: str, coord: str = "User.email", family: str = "f1", split: str = "train") -> QueryRecord:
    return QueryRecord(query_id=query_id, query=query, positive_coordinate=coord, split=split, source="test", family_id=family, quality_score=0.9, owner_type=coord.split(".")[0], field_name=coord.split(".")[1], world_id=f"world-{family}", world_split=split, relevant_coordinates=[coord])


def test_semantic_dedupe_reduces_near_duplicates():
    rows = [_row("1", "find the email field"), _row("2", "find the email field"), _row("3", "read the ticket status", coord="Ticket.status", family="f2")]
    out = semantic_dedupe(rows, threshold=0.90)
    assert len(out) <= 2


def test_leakage_filter_drops_family_with_high_leakage():
    rows = [_row("1", "what is user.email", family="fam-a"), _row("2", "show user.name", coord="User.name", family="fam-a"), _row("3", "what field stores the email address", family="fam-b")]
    out = leakage_filter(rows, canonical_terms={"user.email", "user.name"}, threshold=0.5)
    assert len(out) == 1
    assert out[0].query_id == "3"


def test_ambiguity_filter_drops_vague_queries():
    rows = [_row("1", "show me anything"), _row("2", "what field stores the email address")]
    out = ambiguity_filter(rows)
    assert [r.query_id for r in out] == ["2"]


def test_build_dataset_requires_non_empty_holdout_splits(tmp_path):
    cfg = DatasetBuildConfig(version=99)
    with pytest.raises(ValueError, match="held-out splits"):
        build_dataset(openai_seed_rows=[_row("q1", "what field stores the email address")], corpus=_corpus(), out_dir=tmp_path, schema_hash="abc", cfg=cfg, generation_config={"openai_model": "gpt-4o-mini"})


def test_build_benchmark_suites_includes_curated_release_gate():
    corpus = _corpus()
    rows = [_row("q1", "what field stores the email address", split="test"), _row("q2", "what field gives me the current state", coord="Ticket.status", family="f2", split="test")]
    benches = build_benchmark_suites(rows, corpus)
    assert "curated_challenge_eval" in benches
    assert benches["curated_challenge_eval"]
