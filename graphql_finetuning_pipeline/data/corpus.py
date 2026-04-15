from __future__ import annotations

from graphql_finetuning_pipeline.data.models import CorpusRecord, GraphQLTypeRecord
from graphql_finetuning_pipeline.data.structural_views import structural_views_from_graphql_type


def _desc(desc: str | None) -> str:
    return (desc or "").strip() or "No description provided."


def build_corpus(records: list[GraphQLTypeRecord]) -> list[CorpusRecord]:
    out: list[CorpusRecord] = []
    for t in records:
        for field, (coordinate_text, field_signature_text, field_semantic_text, sdl_snippet_text) in structural_views_from_graphql_type(t):
            coordinate = coordinate_text
            args = ", ".join(f"{a.name}: {a.type_name}" for a in field.args)
            out.append(
                CorpusRecord(
                    doc_id=coordinate,
                    coordinate=coordinate,
                    owner_type=t.type_name,
                    field_name=field.name,
                    return_type=field.type_name,
                    argument_signature=args,
                    description=field.description,
                    short_text=f"Field {coordinate}: {_desc(field.description)}",
                    full_text=field_semantic_text,
                    keywords_text=" | ".join(sorted(set([coordinate.lower(), t.type_name.lower(), field.name.lower(), field.type_name.lower()]))),
                    aliases=[field.name.lower(), field.name.replace("_", " ").lower()],
                    coordinate_text=coordinate_text,
                    field_signature_text=field_signature_text,
                    field_semantic_text=field_semantic_text,
                    sdl_snippet_text=sdl_snippet_text,
                    retrieval_text=field_semantic_text,
                    metadata={
                        "owner_description": t.description,
                        "type_kind": t.type_kind,
                        "arg_names": [a.name for a in field.args],
                        "arg_types": [a.type_name for a in field.args],
                    },
                )
            )
    return out
