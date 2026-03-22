from __future__ import annotations

from graphql_finetuning_pipeline.data.models import CorpusRecord, GraphQLTypeRecord
from graphql_finetuning_pipeline.data.structural_views import structural_views_from_graphql_type


def _desc(desc: str | None) -> str:
    return (desc or "").strip() or "No description provided."


def render_short_text(t: GraphQLTypeRecord) -> str:
    return f"Type {t.type_name}: {_desc(t.description)}"


def render_full_text(t: GraphQLTypeRecord, max_fields: int = 20) -> str:
    fields_part = "; ".join(
        f"{f.name}: {f.type_name}" + (f" ({f.description})" if f.description else "") for f in t.fields[:max_fields]
    )
    args_part = []
    for f in t.fields[:max_fields]:
        for a in f.args[:4]:
            args_part.append(f"{f.name}.{a.name}: {a.type_name}")
    rels = ", ".join(t.interfaces + t.possible_types)
    enum_values = ", ".join(t.enum_values[:20])
    return (
        f"GraphQL type {t.type_name}. Kind: {t.type_kind}. Description: {_desc(t.description)}. "
        f"Fields: {fields_part or 'None'}. "
        f"Representative args: {', '.join(args_part) or 'None'}. "
        f"Related types/interfaces: {rels or 'None'}. Enum values: {enum_values or 'None'}."
    )


def render_keywords_text(t: GraphQLTypeRecord) -> str:
    toks = [t.type_name, t.type_name.lower(), t.type_name.replace("_", " ").lower()]
    toks.extend([f.name.lower() for f in t.fields[:20]])
    toks.extend([f.type_name.lower() for f in t.fields[:20]])
    toks.extend([i.lower() for i in t.interfaces])
    return " | ".join(sorted(set(x for x in toks if x)))


def build_corpus(records: list[GraphQLTypeRecord]) -> list[CorpusRecord]:
    out: list[CorpusRecord] = []
    for t in records:
        type_name_text, field_paths_text, sdl_text = structural_views_from_graphql_type(t)
        out.append(
            CorpusRecord(
                type_id=t.type_id,
                type_name=t.type_name,
                short_text=render_short_text(t),
                full_text=render_full_text(t),
                keywords_text=render_keywords_text(t),
                type_name_text=type_name_text,
                field_paths_text=field_paths_text,
                sdl_text=sdl_text,
                retrieval_text=sdl_text,
                metadata={
                    "type_kind": t.type_kind,
                    "field_count": len(t.fields),
                    "interface_count": len(t.interfaces),
                },
            )
        )
    return out
