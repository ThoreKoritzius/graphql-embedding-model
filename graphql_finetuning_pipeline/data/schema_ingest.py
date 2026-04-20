from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graphql import build_client_schema, build_schema
from graphql.type import GraphQLField, GraphQLNamedType

from graphql_finetuning_pipeline.data.models import FieldArgRecord, FieldRecord, GraphQLTypeRecord


def _unwrap_type_name(type_obj: Any) -> str:
    curr = type_obj
    while hasattr(curr, "of_type") and getattr(curr, "of_type") is not None:
        curr = curr.of_type
    return getattr(curr, "name", str(curr))


def _field_record(field_name: str, field: GraphQLField) -> FieldRecord:
    # Input-object fields (``GraphQLInputField``) do not carry ``.args`` —
    # only output-object fields do. Guard so the same helper can be used
    # for both kinds without crashing on real-world schemas that expose
    # input types.
    raw_args = getattr(field, "args", None) or {}
    args = [
        FieldArgRecord(name=a_name, type_name=_unwrap_type_name(a.type), description=getattr(a, "description", None))
        for a_name, a in raw_args.items()
    ]
    return FieldRecord(
        name=field_name,
        type_name=_unwrap_type_name(field.type),
        description=getattr(field, "description", None),
        args=args,
    )


def _type_record(type_obj: GraphQLNamedType) -> GraphQLTypeRecord:
    fields: list[FieldRecord] = []
    interfaces: list[str] = []
    possible_types: list[str] = []
    enum_values: list[str] = []

    if hasattr(type_obj, "fields") and getattr(type_obj, "fields"):
        fields = [_field_record(n, f) for n, f in type_obj.fields.items()]
    if hasattr(type_obj, "interfaces") and getattr(type_obj, "interfaces"):
        interfaces = [i.name for i in type_obj.interfaces]
    if hasattr(type_obj, "types") and getattr(type_obj, "types"):
        possible_types = [t.name for t in type_obj.types]
    if hasattr(type_obj, "values") and getattr(type_obj, "values"):
        enum_values = list(type_obj.values.keys())

    return GraphQLTypeRecord(
        type_id=type_obj.name,
        type_name=type_obj.name,
        type_kind=type_obj.__class__.__name__,
        description=type_obj.description,
        interfaces=interfaces,
        possible_types=possible_types,
        enum_values=enum_values,
        fields=fields,
    )


def parse_schema(input_path: Path) -> tuple[list[GraphQLTypeRecord], str]:
    raw = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".json":
        payload = json.loads(raw)
        schema_payload = payload.get("data", payload)
        schema = build_client_schema(schema_payload)
    else:
        schema = build_schema(raw)

    records: list[GraphQLTypeRecord] = []
    for name, type_obj in schema.type_map.items():
        if name.startswith("__"):
            continue
        records.append(_type_record(type_obj))

    records.sort(key=lambda x: x.type_name)
    return records, raw
