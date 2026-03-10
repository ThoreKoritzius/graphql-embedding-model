from __future__ import annotations

from pydantic import BaseModel, Field


class FieldArgRecord(BaseModel):
    name: str
    type_name: str
    description: str | None = None


class FieldRecord(BaseModel):
    name: str
    type_name: str
    description: str | None = None
    args: list[FieldArgRecord] = Field(default_factory=list)


class GraphQLTypeRecord(BaseModel):
    type_id: str
    type_name: str
    type_kind: str
    description: str | None = None
    interfaces: list[str] = Field(default_factory=list)
    possible_types: list[str] = Field(default_factory=list)
    enum_values: list[str] = Field(default_factory=list)
    fields: list[FieldRecord] = Field(default_factory=list)


class CorpusRecord(BaseModel):
    type_id: str
    type_name: str
    short_text: str
    full_text: str
    keywords_text: str
    metadata: dict = Field(default_factory=dict)


class QueryRecord(BaseModel):
    query_id: str
    query: str
    target_type_id: str
    split: str
    source: str
    family_id: str
    quality_score: float
    negatives_easy: list[str] = Field(default_factory=list)
    negatives_medium: list[str] = Field(default_factory=list)
    negatives_hard: list[str] = Field(default_factory=list)
