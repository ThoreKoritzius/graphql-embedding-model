from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.synthetic import (
    bootstrap_queries,
    build_benchmark_sets,
    expand_queries,
    quality_filter,
    split_queries,
)


def test_split_by_family_prevents_leakage():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    seed = bootstrap_queries(corpus, rng_seed=1)
    expanded = expand_queries(seed)
    filtered = quality_filter(expanded, type_names={c.type_name for c in corpus}, leakage_threshold=1.0)
    split_rows = split_queries(filtered, seed=1)

    family_to_split = {}
    for row in split_rows:
        prev = family_to_split.get(row.family_id)
        if prev is not None:
            assert prev == row.split
        family_to_split[row.family_id] = row.split


def test_build_benchmark_sets_shapes():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    seed = bootstrap_queries(corpus, rng_seed=3)
    expanded = expand_queries(seed)
    filtered = quality_filter(expanded, type_names={c.type_name for c in corpus}, leakage_threshold=1.0)
    split_rows = split_queries(filtered, seed=2)

    benches = build_benchmark_sets(split_rows, corpus)
    assert "synthetic_holdout" in benches
    assert "realism_seed" in benches
    assert "adversarial_ambiguity" in benches
    assert len(benches["synthetic_holdout"]) == len([x for x in split_rows if x.split == "test"])
