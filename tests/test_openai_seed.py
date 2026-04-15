import json
from pathlib import Path

from graphql_finetuning_pipeline.data.dataset_builder import DatasetBuildConfig, build_dataset
from graphql_finetuning_pipeline.data.openai_seed import OpenAISeedConfig, _extract_json, _validate_items, generate_openai_seed


def test_extract_json_from_markdown_block():
    parsed = _extract_json("```json\n{\"items\": []}\n```")
    assert parsed["items"] == []


def test_validate_items_accepts_single_item_payload():
    parsed = {
        "query": "Who wrote the record?",
        "positive_coordinate": "Post.author",
        "relevant_coordinates": ["Post.author"],
        "negative_coordinates": ["User.name"],
        "difficulty": "medium",
        "intent": "authorship",
    }
    out = _validate_items(parsed=parsed, valid_coordinates={"Post.author", "User.name"}, canonical_terms={"post.author", "user.name", "post", "author", "user", "name"})
    assert len(out) == 1
    assert out[0].positive_coordinate == "Post.author"


def test_mocked_openai_seed_and_dataset_build(tmp_path: Path):
    out_rows, _, corpus = generate_openai_seed(
        out_dir=tmp_path,
        cfg=OpenAISeedConfig(items_per_world=1, world_count=12, worlds_version=1, min_types_per_world=20, max_types_per_world=20),
    )
    assert out_rows
    seeded_rows = []
    for i, row in enumerate(out_rows[:2]):
        seeded_rows.append(row.model_copy(update={"split": "train", "world_split": "train", "query_id": f"train-{i}"}))
        seeded_rows.append(row.model_copy(update={"split": "val", "world_split": "val", "query_id": f"val-{i}"}))
        seeded_rows.append(row.model_copy(update={"split": "test", "world_split": "test", "query_id": f"test-{i}"}))
    manifest = build_dataset(openai_seed_rows=seeded_rows, corpus=corpus, out_dir=tmp_path, schema_hash="abc123", cfg=DatasetBuildConfig(version=1), generation_config={"openai_model": "local-fallback"})
    assert manifest["release_gate_benchmark"] == "curated_challenge_eval"
    assert (tmp_path / "datasets" / "v1" / "benchmarks" / "curated_challenge_eval.jsonl").exists()
