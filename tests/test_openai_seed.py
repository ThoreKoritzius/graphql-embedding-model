import json
from pathlib import Path

from graphql_finetuning_pipeline.data.dataset_builder import DatasetBuildConfig, build_dataset
from graphql_finetuning_pipeline.data.openai_seed import OpenAISeedConfig, _extract_json, _validate_items, generate_openai_seed


def test_extract_json_from_markdown_block():
    text = "```json\n{\"items\": []}\n```"
    parsed = _extract_json(text)
    assert parsed["items"] == []


def test_validate_items_accepts_single_item_payload():
    parsed = {
        "query": "Find unresolved customer escalations quickly",
        "primary_type_id": "world_0000:Ticket",
        "relevant_type_ids": ["world_0000:Ticket", "world_0000:Issue"],
        "difficulty": "medium",
        "negative_type_ids": [],
    }
    out = _validate_items(
        parsed=parsed,
        valid_type_ids={"world_0000:Ticket", "world_0000:Issue"},
        canonical_names={"Ticket", "Issue"},
    )
    assert len(out) == 1
    assert out[0].primary_type_id == "world_0000:Ticket"


def test_mocked_openai_seed_and_dataset_build(tmp_path: Path):
    world_id = "world_0000"
    mock_line = {
        "items": [
            {
                "query": "How can I retrieve this entity details?",
                "primary_type_id": f"{world_id}:Hotel",
                "relevant_type_ids": [f"{world_id}:Hotel", f"{world_id}:Booking"],
                "relation_pair": {"primary": f"{world_id}:Hotel", "bridge": f"{world_id}:Booking"},
                "difficulty": "medium",
                "noise_tags": ["typo"],
                "adversarial_tags": [],
                "negative_type_ids": [f"{world_id}:Room"],
            },
            {
                "query": "Which type supports filtered retrieval for this entity?",
                "primary_type_id": f"{world_id}:Booking",
                "relevant_type_ids": [f"{world_id}:Booking", f"{world_id}:Hotel"],
                "relation_pair": {"primary": f"{world_id}:Booking", "bridge": f"{world_id}:Hotel"},
                "difficulty": "hard",
                "noise_tags": [],
                "adversarial_tags": ["negation"],
                "negative_type_ids": [f"{world_id}:Guest"],
            },
        ]
    }
    mock_path = tmp_path / "mock_openai.jsonl"
    mock_path.write_text(json.dumps(mock_line), encoding="utf-8")

    out_rows, _, corpus = generate_openai_seed(
        out_dir=tmp_path,
        cfg=OpenAISeedConfig(
            items_per_world=2,
            world_count=1,
            worlds_version=1,
            min_types_per_world=20,
            max_types_per_world=20,
        ),
        mock_responses_path=mock_path,
    )
    assert len(out_rows) >= 2
    synthetic_rows = []
    for i, row in enumerate(out_rows[:2]):
        synthetic_rows.append(row.model_copy(update={"split": "train", "world_split": "train", "query_id": f"train-{i}"}))
        synthetic_rows.append(row.model_copy(update={"split": "val", "world_split": "val", "query_id": f"val-{i}"}))
        synthetic_rows.append(row.model_copy(update={"split": "test", "world_split": "test", "query_id": f"test-{i}"}))

    manifest = build_dataset(
        openai_seed_rows=synthetic_rows,
        corpus=corpus,
        out_dir=tmp_path,
        schema_hash="abc123",
        cfg=DatasetBuildConfig(version=1, target_train_min=20, target_train_max=200),
        generation_config={"openai_model": "gpt-4o-mini"},
    )
    assert manifest["version"] == 1
    assert (tmp_path / "datasets" / "v1" / "manifest.json").exists()
    assert (tmp_path / "datasets" / "v1" / "benchmarks" / "realism_eval.jsonl").exists()
