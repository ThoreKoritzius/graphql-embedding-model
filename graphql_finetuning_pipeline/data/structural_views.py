from __future__ import annotations

from hashlib import sha256

from graphql_finetuning_pipeline.data.models import CorpusRecord, GraphQLTypeRecord

VALID_VIEWS = {"typename", "field_paths", "sdl"}


def parse_positive_views(raw: str | None) -> list[str]:
    if not raw:
        return ["typename", "field_paths", "sdl"]
    parts = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not parts:
        return ["typename", "field_paths", "sdl"]
    invalid = [x for x in parts if x not in VALID_VIEWS]
    if invalid:
        raise ValueError(f"Invalid positive view(s): {', '.join(invalid)}. Valid: typename, field_paths, sdl")
    # Deduplicate but keep order.
    out: list[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out


def normalize_primary_retrieval_view(raw: str | None) -> str:
    v = (raw or "sdl").strip().lower()
    if v not in VALID_VIEWS:
        raise ValueError(f"Invalid primary retrieval view: {v}. Valid: typename, field_paths, sdl")
    return v


def render_type_name_text(type_name: str) -> str:
    return type_name.strip()


def render_field_paths_text(type_name: str, fields: list[tuple[str, str]]) -> str:
    if not fields:
        return f"{type_name}->id"
    rows = sorted({(name, ftype) for name, ftype in fields}, key=lambda x: (x[0], x[1]))
    return " | ".join([f"{type_name}->{name}:{ftype}" for name, ftype in rows])


def render_sdl_text(type_name: str, fields: list[tuple[str, str]]) -> str:
    if not fields:
        fields = [("id", "ID!")]
    rows = sorted({(name, ftype) for name, ftype in fields}, key=lambda x: (x[0], x[1]))
    field_lines = "\n".join([f"  {name}: {ftype}" for name, ftype in rows])
    return f"type {type_name} {{\n{field_lines}\n}}"


def structural_views_from_graphql_type(t: GraphQLTypeRecord) -> tuple[str, str, str]:
    fields = [(f.name, f.type_name or "String") for f in t.fields]
    type_name_text = render_type_name_text(t.type_name)
    field_paths_text = render_field_paths_text(t.type_name, fields)
    sdl_text = render_sdl_text(t.type_name, fields)
    return type_name_text, field_paths_text, sdl_text


def structural_views_from_corpus_record(c: CorpusRecord) -> tuple[str, str, str]:
    type_name_text = c.type_name_text or render_type_name_text(c.type_name)

    if c.field_paths_text:
        field_paths_text = c.field_paths_text
    else:
        field_names = c.metadata.get("fields", [])
        fields = [(name, "String") for name in field_names] if field_names else [("id", "ID!")]
        field_paths_text = render_field_paths_text(c.type_name, fields)

    if c.sdl_text:
        sdl_text = c.sdl_text
    else:
        field_names = c.metadata.get("fields", [])
        fields = [(name, "String") for name in field_names] if field_names else [("id", "ID!")]
        sdl_text = render_sdl_text(c.type_name, fields)

    return type_name_text, field_paths_text, sdl_text


def get_view_text(c: CorpusRecord, view: str) -> str:
    view = normalize_primary_retrieval_view(view)
    type_name_text, field_paths_text, sdl_text = structural_views_from_corpus_record(c)
    if view == "typename":
        return type_name_text
    if view == "field_paths":
        return field_paths_text
    return sdl_text


def get_positive_texts(c: CorpusRecord, views: list[str]) -> list[str]:
    type_name_text, field_paths_text, sdl_text = structural_views_from_corpus_record(c)
    mapping = {
        "typename": type_name_text,
        "field_paths": field_paths_text,
        "sdl": sdl_text,
    }
    out: list[str] = []
    for v in views:
        txt = mapping.get(v, "").strip()
        if txt:
            out.append(txt)
    return out


def ensure_view_available(corpus_rows: list[CorpusRecord], view: str) -> None:
    v = normalize_primary_retrieval_view(view)
    missing = []
    for c in corpus_rows:
        if v == "typename" and not (c.type_name_text or c.type_name):
            missing.append(c.type_id)
        elif v == "field_paths" and not c.field_paths_text and not c.metadata.get("fields"):
            missing.append(c.type_id)
        elif v == "sdl" and not c.sdl_text and not c.metadata.get("fields"):
            missing.append(c.type_id)
        if len(missing) >= 10:
            break
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"Corpus is missing required '{v}' view for one or more types (sample: {sample}). "
            "Run `graphft backfill-structural-corpus --corpus <in> --out <out>` and retry."
        )


def corpus_structural_hash(corpus_rows: list[CorpusRecord]) -> str:
    h = sha256()
    for c in sorted(corpus_rows, key=lambda x: x.type_id):
        a, b, d = structural_views_from_corpus_record(c)
        h.update(f"{c.type_id}\n{a}\n{b}\n{d}\n".encode("utf-8"))
    return h.hexdigest()
