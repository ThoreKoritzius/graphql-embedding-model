import json
from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.dataset_builder import DatasetBuildConfig, build_dataset
from graphql_finetuning_pipeline.data.openai_seed import OpenAISeedConfig, _extract_json, generate_openai_seed
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema


def test_extract_json_from_markdown_block():
    text = "```json\n{\"items\": []}\n```"
    parsed = _extract_json(text)
    assert parsed["items"] == []


def test_mocked_openai_seed_and_dataset_build(tmp_path: Path):
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)[:2]

    mock_line = {
        "items": [
            {
                "query": "How can I retrieve this entity details from GraphQL?",
                "target_type_id": corpus[0].type_id,
                "intent": "lookup",
                "difficulty": "medium",
                "confuser_type_ids": [corpus[1].type_id],
                "rationale_tags": ["semantic"],
            },
            {
                "query": "Which type supports filtered retrieval for this entity?",
                "target_type_id": corpus[0].type_id,
                "intent": "filtering",
                "difficulty": "hard",
                "confuser_type_ids": [corpus[1].type_id],
                "rationale_tags": ["ambiguous"],
            },
        ]
    }
    mock_path = tmp_path / "mock_openai.jsonl"
    mock_path.write_text("\n".join([json.dumps(mock_line), json.dumps(mock_line)]), encoding="utf-8")

    out_rows, _ = generate_openai_seed(
        corpus=corpus,
        out_dir=tmp_path,
        cfg=OpenAISeedConfig(items_per_type=2),
        mock_responses_path=mock_path,
    )
    assert len(out_rows) >= 2

    manifest = build_dataset(
        openai_seed_rows=out_rows,
        corpus=corpus,
        out_dir=tmp_path,
        schema_hash="abc123",
        cfg=DatasetBuildConfig(version=1, target_train_min=4, target_train_max=20),
        generation_config={"openai_model": "gpt-4o-mini"},
    )
    assert manifest["version"] == 1
    assert (tmp_path / "datasets" / "v1" / "manifest.json").exists()
    assert (tmp_path / "datasets" / "v1" / "benchmarks" / "realism_eval.jsonl").exists()
