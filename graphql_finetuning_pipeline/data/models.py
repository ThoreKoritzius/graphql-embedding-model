from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    model_config = ConfigDict(populate_by_name=True)

    doc_id: str
    coordinate: str
    owner_type: str
    field_name: str
    return_type: str
    short_text: str
    full_text: str
    keywords_text: str
    argument_signature: str = ""
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    path_to_root: list[str] = Field(default_factory=list)
    coordinate_text: str = ""
    field_signature_text: str = ""
    field_semantic_text: str = ""
    sdl_snippet_text: str = ""
    retrieval_text: str = ""
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy(cls, data):
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "doc_id" not in payload and "type_id" in payload:
            payload["doc_id"] = payload["type_id"]
        if "coordinate" not in payload:
            type_name = payload.get("type_name") or payload.get("owner_type") or payload.get("coordinate") or payload.get("doc_id", "")
            field_name = payload.get("field_name") or payload.get("metadata", {}).get("field_name")
            payload["coordinate"] = f"{type_name}.{field_name}" if field_name else type_name
        if "owner_type" not in payload:
            payload["owner_type"] = payload.get("type_name") or payload.get("coordinate", "").split(".", 1)[0]
        if "field_name" not in payload:
            payload["field_name"] = payload.get("metadata", {}).get("field_name") or payload.get("coordinate", "").split(".", 1)[-1]
        if "return_type" not in payload:
            payload["return_type"] = payload.get("metadata", {}).get("return_type", "String")
        if "coordinate_text" not in payload:
            payload["coordinate_text"] = payload.get("type_name_text", payload.get("coordinate", ""))
        if "field_signature_text" not in payload:
            payload["field_signature_text"] = payload.get("field_paths_text", "")
        if "field_semantic_text" not in payload:
            payload["field_semantic_text"] = payload.get("full_text", "")
        if "sdl_snippet_text" not in payload:
            payload["sdl_snippet_text"] = payload.get("sdl_text", "")
        if "retrieval_text" not in payload:
            payload["retrieval_text"] = payload.get("sdl_text") or payload.get("field_semantic_text", "")
        return payload

    @property
    def type_id(self) -> str:
        return self.doc_id

    @property
    def type_name(self) -> str:
        return self.owner_type

    @property
    def type_name_text(self) -> str:
        return self.coordinate_text

    @property
    def field_paths_text(self) -> str:
        return self.field_signature_text

    @property
    def sdl_text(self) -> str:
        return self.sdl_snippet_text


class QueryRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query_id: str
    query: str
    positive_coordinate: str
    split: str
    source: str
    family_id: str
    quality_score: float
    negative_coordinates: list[str] = Field(default_factory=list)
    relevant_coordinates: list[str] = Field(default_factory=list)
    owner_type: str | None = None
    field_name: str | None = None
    world_id: str | None = None
    domain: str | None = None
    world_split: str | None = None
    difficulty: str | None = None
    intent: str | None = None
    confuser_tags: list[str] = Field(default_factory=list)
    adversarial_tags: list[str] = Field(default_factory=list)
    noise_tags: list[str] = Field(default_factory=list)
    rationale_tags: list[str] = Field(default_factory=list)
    confuser_coordinates: list[str] = Field(default_factory=list)
    negatives_easy: list[str] = Field(default_factory=list)
    negatives_medium: list[str] = Field(default_factory=list)
    negatives_hard: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy(cls, data):
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "positive_coordinate" not in payload and "target_type_id" in payload:
            payload["positive_coordinate"] = payload["target_type_id"]
        if "relevant_coordinates" not in payload:
            payload["relevant_coordinates"] = payload.get("relevant_type_ids", [])
        if "negative_coordinates" not in payload:
            payload["negative_coordinates"] = payload.get("negative_type_ids", [])
        if "confuser_coordinates" not in payload:
            payload["confuser_coordinates"] = payload.get("confuser_type_ids", [])
        if "owner_type" not in payload:
            coord = payload.get("positive_coordinate", "")
            payload["owner_type"] = coord.split(".", 1)[0] if "." in coord else payload.get("primary_type_id")
        if "field_name" not in payload:
            coord = payload.get("positive_coordinate", "")
            payload["field_name"] = coord.split(".", 1)[1] if "." in coord else None
        return payload

    @property
    def target_type_id(self) -> str:
        return self.positive_coordinate

    @property
    def relevant_type_ids(self) -> list[str]:
        return self.relevant_coordinates

    @property
    def negative_type_ids(self) -> list[str]:
        return self.negative_coordinates

    @property
    def confuser_type_ids(self) -> list[str]:
        return self.confuser_coordinates

    @property
    def primary_type_id(self) -> str | None:
        return self.positive_coordinate

    @property
    def relation_pair(self) -> dict[str, str]:
        return {}
