from __future__ import annotations

from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.models import CorpusRecord, GraphQLTypeRecord
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.structural_views import get_view_text, normalize_primary_retrieval_view, structural_views_from_corpus_record


def load_world_type_lookup(worlds_dir: Path) -> dict[str, GraphQLTypeRecord]:
    if not worlds_dir.exists():
        raise FileNotFoundError(f"Worlds directory does not exist: {worlds_dir}")
    out: dict[str, GraphQLTypeRecord] = {}
    for world_dir in sorted(worlds_dir.glob("world_*")):
        schema_path = world_dir / "schema.graphql"
        if not schema_path.exists():
            continue
        records, _ = parse_schema(schema_path)
        for rec in records:
            out[f"{world_dir.name}:{rec.type_name}"] = rec
    return out


def backfill_structural_views(corpus_rows: list[CorpusRecord], *, primary_retrieval_view: str = "semantic", world_type_lookup: dict[str, GraphQLTypeRecord] | None = None) -> list[CorpusRecord]:
    primary = normalize_primary_retrieval_view(primary_retrieval_view)
    lookup = world_type_lookup or {}
    rebuilt_by_doc: dict[str, CorpusRecord] = {}
    if lookup:
        rebuilt = []
        for prefix, rec in lookup.items():
            world_id = prefix.split(":", 1)[0]
            for row in build_corpus([rec]):
                rebuilt.append(row.model_copy(update={"doc_id": f"{world_id}:{row.coordinate}"}))
        rebuilt_by_doc = {r.doc_id: r for r in rebuilt}

    out: list[CorpusRecord] = []
    for row in corpus_rows:
        updated = rebuilt_by_doc.get(row.doc_id, row)
        coordinate_text, field_signature_text, field_semantic_text, sdl_snippet_text = structural_views_from_corpus_record(updated)
        updated = updated.model_copy(update={"coordinate_text": coordinate_text, "field_signature_text": field_signature_text, "field_semantic_text": field_semantic_text, "sdl_snippet_text": sdl_snippet_text})
        out.append(updated.model_copy(update={"retrieval_text": get_view_text(updated, primary)}))
    return out
