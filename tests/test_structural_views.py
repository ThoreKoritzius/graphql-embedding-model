from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.corpus_backfill import backfill_structural_views
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.structural_views import get_positive_texts, render_sdl_text


def test_render_sdl_text_is_stable():
    s1 = render_sdl_text("User", [("email", "String!"), ("id", "ID!")])
    s2 = render_sdl_text("User", [("id", "ID!"), ("email", "String!")])
    assert s1 == s2
    assert s1.startswith("type User")


def test_backfill_structural_views_populates_required_fields():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    for c in corpus:
        c.type_name_text = ""
        c.field_paths_text = ""
        c.sdl_text = ""
        c.retrieval_text = ""

    out = backfill_structural_views(corpus, primary_retrieval_view="sdl")
    assert len(out) == len(corpus)
    assert all(r.type_name_text for r in out)
    assert all(r.field_paths_text for r in out)
    assert all(r.sdl_text.startswith("type ") for r in out)
    assert all(r.retrieval_text.startswith("type ") for r in out)


def test_get_positive_texts_all_three_views():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    first = corpus[0]
    docs = get_positive_texts(first, ["typename", "field_paths", "sdl"])
    assert len(docs) == 3
