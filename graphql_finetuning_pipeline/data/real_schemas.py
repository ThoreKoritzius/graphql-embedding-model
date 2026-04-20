"""Ingest real-world GraphQL SDLs as additional "worlds" for training.

The seed generator was previously fed only synthetic template worlds. This
produces a model that can crush templated holdouts but has never seen the
naming quirks, deep nesting, unions, interfaces, or connection patterns
that appear in real customer schemas. Mixing real SDLs in closes that gap.

Each SDL directory becomes one or more worlds (one per file). Real worlds
are tagged ``source="real-schema"`` in the world manifest so downstream
code can split, weight, or report on them separately.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from graphql_finetuning_pipeline.data.corpus import build_corpus
from graphql_finetuning_pipeline.data.models import CorpusRecord
from graphql_finetuning_pipeline.data.schema_ingest import parse_schema
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_json, write_jsonl


@dataclass
class RealSchemaConfig:
    version: int = 1
    real_split_overrides: dict[str, str] | None = None  # filename -> "train"|"val"|"test"


def _infer_domain(sdl_path: Path) -> str:
    stem = sdl_path.stem.lower()
    for key in ("ecommerce", "shop", "shopify", "stripe", "fintech", "github", "hotel", "travel", "healthcare", "support"):
        if key in stem:
            return key
    return "real"


def _field_catalog_from_corpus(corpus: list[CorpusRecord]) -> list[dict]:
    return [
        {
            "doc_id": c.doc_id,
            "coordinate": c.coordinate,
            "owner_type": c.owner_type,
            "field_name": c.field_name,
            "return_type": c.return_type,
            "description": c.description,
            # Owner descriptions power owner-type disambiguation in the
            # GPT prompt. parse_schema → build_corpus stores them in
            # CorpusRecord.metadata["owner_description"]; surfacing here
            # is what makes real SDLs (GHES, Shopify) usable in the
            # competition-set prompt.
            "owner_description": (c.metadata or {}).get("owner_description") or "",
            "aliases": c.aliases,
            "path_to_root": c.path_to_root,
        }
        for c in corpus
    ]


def ingest_real_schemas(sdl_dir: Path, out_dir: Path, cfg: RealSchemaConfig) -> tuple[list[dict], list[CorpusRecord]]:
    """Parse every ``*.graphql`` / ``*.json`` (introspection result) under
    ``sdl_dir`` and emit world entries + corpus rows matching the existing
    synthetic-world layout.

    Returns (world_metadata, corpus_rows). Caller is responsible for merging
    into the global world manifest and corpus.
    """
    if not sdl_dir.exists() or not sdl_dir.is_dir():
        raise FileNotFoundError(f"Real-schema directory not found: {sdl_dir}")

    worlds_dir = out_dir / "worlds" / f"v{cfg.version}"
    ensure_dir(worlds_dir)
    overrides = cfg.real_split_overrides or {}

    world_rows: list[dict] = []
    all_corpus_rows: list[CorpusRecord] = []

    sdl_paths = sorted(list(sdl_dir.glob("*.graphql")) + list(sdl_dir.glob("*.json")))
    if not sdl_paths:
        raise FileNotFoundError(f"No .graphql or .json schemas found under {sdl_dir}")

    for sdl_path in sdl_paths:
        records, raw = parse_schema(sdl_path)
        schema_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        world_id = f"real_{sdl_path.stem}_{schema_hash}"
        domain = _infer_domain(sdl_path)
        split = overrides.get(sdl_path.name, overrides.get(sdl_path.stem, "test"))

        world_dir = worlds_dir / world_id
        ensure_dir(world_dir)
        (world_dir / "schema.graphql").write_text(raw, encoding="utf-8")

        base_corpus = build_corpus(records)
        world_corpus: list[CorpusRecord] = []
        for c in base_corpus:
            md = dict(c.metadata)
            md.update({"world_id": world_id, "domain": domain, "source": "real-schema", "schema_file": sdl_path.name})
            world_corpus.append(c.model_copy(update={"doc_id": f"{world_id}:{c.coordinate}", "metadata": md}))

        all_corpus_rows.extend(world_corpus)

        field_catalog = _field_catalog_from_corpus(world_corpus)
        write_json(world_dir / "field_catalog.json", {"world_id": world_id, "domain": domain, "fields": field_catalog})
        world_rows.append(
            {
                "world_id": world_id,
                "domain": domain,
                "type_count": len({c.owner_type for c in world_corpus}),
                "field_count": len(world_corpus),
                "path": str(world_dir.resolve()),
                "source": "real-schema",
                "split": split,
                "schema_file": sdl_path.name,
            }
        )

    write_jsonl(worlds_dir / "real_worlds.jsonl", world_rows)
    return world_rows, all_corpus_rows
