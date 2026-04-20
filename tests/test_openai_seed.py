import json
from pathlib import Path

from graphql_finetuning_pipeline.data.dataset_builder import DatasetBuildConfig, build_dataset
from graphql_finetuning_pipeline.data.openai_seed import (
    OpenAISeedConfig,
    SeedItem,
    _ambiguity_map,
    _extract_json,
    _validate_items,
    generate_openai_seed,
)


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


def test_local_fallback_seed_and_dataset_build(tmp_path: Path):
    # No OPENAI_API_KEY in tests — we have to opt in to the local fallback
    # explicitly so the pipeline cannot silently ship mislabeled rows.
    out_rows, _, corpus = generate_openai_seed(
        out_dir=tmp_path,
        cfg=OpenAISeedConfig(items_per_world=1, world_count=12, worlds_version=1, min_types_per_world=20, max_types_per_world=20, allow_local_fallback=True),
    )
    assert out_rows
    # Every row must be stamped as local-fallback with the reduced quality
    # score — this is the contract the dataset sanity report relies on.
    assert all(r.source == "local-fallback" and r.quality_score == 0.3 for r in out_rows)
    seeded_rows = []
    for i, row in enumerate(out_rows[:2]):
        seeded_rows.append(row.model_copy(update={"split": "train", "world_split": "train", "query_id": f"train-{i}"}))
        seeded_rows.append(row.model_copy(update={"split": "val", "world_split": "val", "query_id": f"val-{i}"}))
        seeded_rows.append(row.model_copy(update={"split": "test", "world_split": "test", "query_id": f"test-{i}"}))
    manifest = build_dataset(openai_seed_rows=seeded_rows, corpus=corpus, out_dir=tmp_path, schema_hash="abc123", cfg=DatasetBuildConfig(version=1), generation_config={"openai_model": "local-fallback"})
    assert manifest["release_gate_benchmark"] == "curated_challenge_eval"
    assert (tmp_path / "datasets" / "v1" / "benchmarks" / "curated_challenge_eval.jsonl").exists()
    assert (tmp_path / "datasets" / "v1" / "sanity_report.json").exists()
    assert (tmp_path / "datasets" / "v1" / "sanity_sample.jsonl").exists()


def test_generate_seed_without_key_fails_without_opt_in(tmp_path: Path):
    import pytest
    with pytest.raises(RuntimeError, match="allow-local-fallback"):
        generate_openai_seed(
            out_dir=tmp_path,
            cfg=OpenAISeedConfig(items_per_world=1, world_count=1, worlds_version=1, min_types_per_world=20, max_types_per_world=20),
        )


def test_ambiguity_map_keeps_only_multi_owner_names():
    catalog = [
        {"coordinate": "Post.author", "owner_type": "Post", "field_name": "author", "return_type": "User"},
        {"coordinate": "Issue.author", "owner_type": "Issue", "field_name": "author", "return_type": "User"},
        {"coordinate": "Post.title", "owner_type": "Post", "field_name": "title", "return_type": "String"},
    ]
    amb = _ambiguity_map(catalog)
    # "author" is on 2 owners -> kept; "title" only on Post -> dropped.
    assert set(amb.keys()) == {"author"}
    assert {f["coordinate"] for f in amb["author"]} == {"Post.author", "Issue.author"}


def test_adversarial_stratify_buckets():
    """Stratify downsamples toward the requested bucket mix."""
    from graphql_finetuning_pipeline.data.adversarial_mining import ScoredItem, stratify, bucket_counts

    def mk(bucket: str, i: int) -> ScoredItem:
        item = SeedItem(
            query=f"q{i}", positive_coordinate="X.y", difficulty="hard", intent="lookup",
        )
        return ScoredItem(item=item, bucket=bucket, base_top1_correct=(bucket != "hard"),
                          base_target_rank=(1 if bucket == "hard" else 0),
                          base_margin=(0.01 if bucket == "medium" else 0.2),
                          base_top1_wrong_coord=("X.z" if bucket == "hard" else None))

    scored = (
        [mk("hard", i) for i in range(10)]
        + [mk("medium", i) for i in range(20)]
        + [mk("easy", i) for i in range(30)]
    )
    kept = stratify(scored, target_hard_fraction=0.55, target_medium_fraction=0.30)
    counts = bucket_counts(kept)
    total = sum(counts.values())
    assert total > 0
    # Hard bucket is the scarcest given 55% target → total bounded by 10/0.55 ≈ 18
    assert total <= 18
    # Ratios within the bucket mix should be close to the targets.
    hard_frac = counts["hard"] / total
    medium_frac = counts["medium"] / total
    assert 0.45 <= hard_frac <= 0.65
    assert 0.20 <= medium_frac <= 0.35
