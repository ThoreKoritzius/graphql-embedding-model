from pathlib import Path

from graphql_finetuning_pipeline.data.schema_ingest import parse_schema


def test_parse_schema_handles_core_types():
    rows, _ = parse_schema(Path("examples/schema.graphql"))
    names = {r.type_name for r in rows}
    assert "User" in names
    assert "Order" in names
    assert "OrderStatus" in names

    user = [r for r in rows if r.type_name == "User"][0]
    assert user.fields
    assert any(f.name == "orders" for f in user.fields)
