from __future__ import annotations

from pathlib import Path

from graphql_finetuning_pipeline.data.models import CorpusRecord, GraphQLTypeRecord
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.data.structural_views import (
    get_view_text,
    normalize_primary_retrieval_view,
    structural_views_from_corpus_record,
    structural_views_from_graphql_type,
)


def load_world_type_lookup(worlds_dir: Path) -> dict[str, GraphQLTypeRecord]:
    if not worlds_dir.exists():
        raise FileNotFoundError(f"Worlds directory does not exist: {worlds_dir}")
    out: dict[str, GraphQLTypeRecord] = {}
    for world_dir in sorted(worlds_dir.glob("world_*")):
        schema_path = world_dir / "schema.graphql"
        if not schema_path.exists():
            continue
        records, _ = parse_schema(schema_path)
        world_id = world_dir.name
        for rec in records:
            out[f"{world_id}:{rec.type_name}"] = rec
    return out


def backfill_structural_views(
    corpus_rows: list[CorpusRecord],
    *,
    primary_retrieval_view: str = "sdl",
    world_type_lookup: dict[str, GraphQLTypeRecord] | None = None,
) -> list[CorpusRecord]:
    primary = normalize_primary_retrieval_view(primary_retrieval_view)
    lookup = world_type_lookup or {}
    out: list[CorpusRecord] = []

    for row in corpus_rows:
        rec = lookup.get(row.type_id)
        if rec is not None:
            type_name_text, field_paths_text, sdl_text = structural_views_from_graphql_type(rec)
        else:
            type_name_text, field_paths_text, sdl_text = structural_views_from_corpus_record(row)

        updated = row.model_copy(
            update={
                "type_name_text": type_name_text,
                "field_paths_text": field_paths_text,
                "sdl_text": sdl_text,
            }
        )
        retrieval_text = get_view_text(updated, primary)
        out.append(updated.model_copy(update={"retrieval_text": retrieval_text}))

    return out

