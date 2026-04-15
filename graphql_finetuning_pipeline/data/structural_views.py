from __future__ import annotations

from hashlib import sha256

from graphql_finetuning_pipeline.data.models import CorpusRecord, FieldRecord, GraphQLTypeRecord

VALID_VIEWS = {"coordinate", "signature", "semantic", "sdl"}
LEGACY_VIEW_MAP = {
    "typename": "coordinate",
    "field_paths": "signature",
    "sdl": "sdl",
    "semantic": "semantic",
    "coordinate": "coordinate",
    "signature": "signature",
}


def parse_positive_views(raw: str | None) -> list[str]:
    if not raw:
        return ["coordinate", "signature", "semantic", "sdl"]
    parts = [normalize_primary_retrieval_view(x) for x in raw.split(",") if x.strip()]
    out: list[str] = []
    for p in parts:
        if p not in out:
            out.append(p)
    return out or ["coordinate", "signature", "semantic", "sdl"]


def normalize_primary_retrieval_view(raw: str | None) -> str:
    v = (raw or "semantic").strip().lower()
    v = LEGACY_VIEW_MAP.get(v, v)
    if v not in VALID_VIEWS:
        raise ValueError(f"Invalid primary retrieval view: {v}. Valid: coordinate, signature, semantic, sdl")
    return v


def render_coordinate_text(owner_type: str, field_name: str) -> str:
    return f"{owner_type}.{field_name}".strip(".")


def render_field_signature_text(owner_type: str, field: FieldRecord) -> str:
    args = ", ".join(f"{a.name}: {a.type_name}" for a in field.args)
    arg_part = f"({args})" if args else ""
    return f"{owner_type}.{field.name}{arg_part}: {field.type_name}"


def render_field_semantic_text(owner_type: str, field: FieldRecord, owner_description: str | None = None) -> str:
    aliases = [field.name.replace("_", " "), field.name.lower()]
    arg_names = ", ".join(a.name for a in field.args) or "none"
    arg_types = ", ".join(a.type_name for a in field.args) or "none"
    return (
        f"GraphQL field {owner_type}.{field.name}. "
        f"Owner type: {owner_type}. "
        f"Returns: {field.type_name}. "
        f"Arguments: names {arg_names}; types {arg_types}. "
        f"Field description: {(field.description or '').strip() or 'No description provided.'} "
        f"Owner description: {(owner_description or '').strip() or 'No description provided.'} "
        f"Aliases: {', '.join(sorted(set(a for a in aliases if a)))}."
    )


def render_sdl_snippet_text(owner_type: str, field: FieldRecord) -> str:
    args = ", ".join(f"{a.name}: {a.type_name}" for a in field.args)
    arg_part = f"({args})" if args else ""
    return f"type {owner_type} {{\n  {field.name}{arg_part}: {field.type_name}\n}}"


def structural_views_from_field(owner_type: str, field: FieldRecord, owner_description: str | None = None) -> tuple[str, str, str, str]:
    return (
        render_coordinate_text(owner_type, field.name),
        render_field_signature_text(owner_type, field),
        render_field_semantic_text(owner_type, field, owner_description=owner_description),
        render_sdl_snippet_text(owner_type, field),
    )


def structural_views_from_graphql_type(t: GraphQLTypeRecord) -> list[tuple[FieldRecord, tuple[str, str, str, str]]]:
    return [(field, structural_views_from_field(t.type_name, field, owner_description=t.description)) for field in t.fields]


def structural_views_from_corpus_record(c: CorpusRecord) -> tuple[str, str, str, str]:
    coordinate_text = c.coordinate_text or c.coordinate
    field_signature_text = c.field_signature_text or c.coordinate
    field_semantic_text = c.field_semantic_text or c.full_text
    sdl_snippet_text = c.sdl_snippet_text or c.retrieval_text
    return coordinate_text, field_signature_text, field_semantic_text, sdl_snippet_text


def get_view_text(c: CorpusRecord, view: str) -> str:
    view = normalize_primary_retrieval_view(view)
    coordinate_text, field_signature_text, field_semantic_text, sdl_snippet_text = structural_views_from_corpus_record(c)
    if view == "coordinate":
        return coordinate_text
    if view == "signature":
        return field_signature_text
    if view == "sdl":
        return sdl_snippet_text
    return field_semantic_text


def get_positive_texts(c: CorpusRecord, views: list[str]) -> list[str]:
    coordinate_text, field_signature_text, field_semantic_text, sdl_snippet_text = structural_views_from_corpus_record(c)
    mapping = {
        "coordinate": coordinate_text,
        "signature": field_signature_text,
        "semantic": field_semantic_text,
        "sdl": sdl_snippet_text,
    }
    out: list[str] = []
    for v in views:
        txt = mapping.get(normalize_primary_retrieval_view(v), "").strip()
        if txt:
            out.append(txt)
    return out


def ensure_view_available(corpus_rows: list[CorpusRecord], view: str) -> None:
    v = normalize_primary_retrieval_view(view)
    missing = []
    for c in corpus_rows:
        txt = get_view_text(c, v).strip()
        if not txt:
            missing.append(c.doc_id)
        if len(missing) >= 10:
            break
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(
            f"Corpus is missing required '{v}' view for one or more coordinates (sample: {sample}). "
            "Run `graphft backfill-structural-corpus --corpus <in> --out <out>` and retry."
        )


def corpus_structural_hash(corpus_rows: list[CorpusRecord]) -> str:
    h = sha256()
    for c in sorted(corpus_rows, key=lambda x: x.doc_id):
        a, b, d, e = structural_views_from_corpus_record(c)
        h.update(f"{c.doc_id}\n{a}\n{b}\n{d}\n{e}\n".encode("utf-8"))
    return h.hexdigest()
