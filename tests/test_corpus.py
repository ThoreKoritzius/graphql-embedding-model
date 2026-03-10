from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema


def test_corpus_renderer_is_deterministic():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    c1 = build_corpus(rows)
    c2 = build_corpus(rows)
    assert [x.full_text for x in c1] == [x.full_text for x in c2]
