from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.corpus_backfill import backfill_structural_views
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.structural_views import get_positive_texts, structural_views_from_corpus_record


def test_field_structural_views_are_present():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    first = corpus[0]
    coordinate, signature, semantic, sdl = structural_views_from_corpus_record(first)
    assert "." in coordinate
    assert ":" in signature
    assert semantic.startswith("GraphQL field")
    assert sdl.startswith("type ")


def test_backfill_structural_views_populates_required_fields():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    blank = [c.model_copy(update={"coordinate_text": "", "field_signature_text": "", "field_semantic_text": "", "sdl_snippet_text": "", "retrieval_text": ""}) for c in corpus]
    out = backfill_structural_views(blank, primary_retrieval_view="semantic")
    assert len(out) == len(corpus)
    assert all(r.coordinate_text for r in out)
    assert all(r.field_signature_text for r in out)
    assert all(r.field_semantic_text.startswith("GraphQL field") for r in out)
    assert all(r.retrieval_text for r in out)


def test_get_positive_texts_all_views():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    corpus = build_corpus(rows)
    docs = get_positive_texts(corpus[0], ["coordinate", "signature", "semantic", "sdl"])
    assert len(docs) == 4
