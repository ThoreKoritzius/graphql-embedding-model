from graphql_finetuning_pipeline.data.dataset_builder import build_benchmark_suites
from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord


def test_build_benchmark_sets_shapes():
    corpus = [CorpusRecord(doc_id="Post.author", coordinate="Post.author", owner_type="Post", field_name="author", return_type="User", short_text="", full_text="GraphQL field Post.author", keywords_text="", coordinate_text="Post.author", field_signature_text="Post.author: User", field_semantic_text="GraphQL field Post.author", sdl_snippet_text="type Post {\n  author: User\n}", retrieval_text="GraphQL field Post.author", metadata={"world_id": "w1", "domain": "content"})]
    rows = [QueryRecord(query_id="q1", query="who wrote the post", positive_coordinate="Post.author", split="test", source="openai-world-seed", family_id="f1", quality_score=1.0, owner_type="Post", field_name="author", world_id="w1", world_split="test", intent="authorship")]
    benches = build_benchmark_suites(rows, corpus)
    assert "synthetic_holdout" in benches
    assert "curated_challenge_eval" in benches
    assert len(benches["synthetic_holdout"]) == 1
