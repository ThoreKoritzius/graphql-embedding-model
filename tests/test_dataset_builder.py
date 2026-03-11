from graphql_finetuning_pipeline.data.dataset_builder import ambiguity_filter, leakage_filter, semantic_dedupe
from graphql_finetuning_pipeline.data.models import QueryRecord


def _row(query_id: str, query: str, family: str = "f1") -> QueryRecord:
    return QueryRecord(
        query_id=query_id,
        query=query,
        target_type_id="User",
        split="train",
        source="test",
        family_id=family,
        quality_score=0.9,
        intent="lookup",
        difficulty="medium",
        world_id=f"world-{family}",
        relevant_type_ids=["User"],
    )


def test_semantic_dedupe_reduces_near_duplicates():
    rows = [_row("1", "find user by id"), _row("2", "find user by id"), _row("3", "lookup order by id", "f2")]
    out = semantic_dedupe(rows, threshold=0.90)
    assert len(out) <= 2


def test_leakage_filter_drops_family_with_high_leakage():
    rows = [_row("1", "what is User type", "fam-a"), _row("2", "show User fields", "fam-a"), _row("3", "fetch account details", "fam-b")]
    out = leakage_filter(rows, canonical_names={"User"}, threshold=0.5)
    assert len(out) == 1
    assert out[0].query_id == "3"


def test_ambiguity_filter_drops_vague_queries():
    rows = [_row("1", "show me anything"), _row("2", "retrieve user billing details")]
    out = ambiguity_filter(rows)
    assert len(out) == 1
    assert out[0].query_id == "2"
