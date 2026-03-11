from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.data.schema_worlds import WorldConfig, generate_schema_worlds
from graphql_finetuning_pipeline.utils.io import ensure_dir, read_json, write_jsonl

DIFFICULTIES = ["easy", "medium", "hard"]


class SeedItem(BaseModel):
    query: str
    primary_type_id: str
    relevant_type_ids: list[str] = Field(default_factory=list)
    relation_pair: dict[str, str] | None = None
    difficulty: str
    noise_tags: list[str] = Field(default_factory=list)
    adversarial_tags: list[str] = Field(default_factory=list)
    negative_type_ids: list[str] = Field(default_factory=list)


class SeedBatch(BaseModel):
    items: list[SeedItem]


@dataclass
class OpenAISeedConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.8
    max_tokens: int = 2400
    retries: int = 3
    timeout_seconds: int = 60
    items_per_world: int = 240
    worlds_version: int = 1
    world_count: int = 60
    min_types_per_world: int = 20
    max_types_per_world: int = 36


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def _system_prompt() -> str:
    return (
        "You generate realistic user questions for retrieving relevant GraphQL entity types. "
        "Users do not know schema names. Return strict JSON with top-level key 'items'."
    )


def _user_prompt(world_id: str, domain: str, catalog: dict[str, dict], cfg: OpenAISeedConfig) -> str:
    type_rows = []
    for type_name, meta in list(catalog.items())[: min(len(catalog), 80)]:
        type_rows.append(
            {
                "type_id": f"{world_id}:{type_name}",
                "aliases": meta.get("aliases", []),
                "neighbors": [f"{world_id}:{n}" for n in meta.get("neighbors", [])],
                "semantic_tags": meta.get("semantic_tags", []),
            }
        )

    return (
        f"World id: {world_id}\n"
        f"Domain: {domain}\n"
        f"Generate {cfg.items_per_world} user questions.\n"
        "Constraints:\n"
        "- User questions must not include canonical type names.\n"
        "- Questions should be realistic and diverse (goals, filters, comparisons, troubleshooting, multi-hop).\n"
        "- Include hard/confusing/adversarial/noisy examples by default.\n"
        "- Output difficulty in easy|medium|hard.\n"
        "- Output 2-5 relevant_type_ids where possible.\n"
        "- primary_type_id must be in relevant_type_ids.\n"
        "Schema type catalog JSON:\n"
        f"{json.dumps(type_rows, ensure_ascii=True)}\n"
        "Output schema:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "query": "...",\n'
        '      "primary_type_id": "world_xxxx:Type",\n'
        '      "relevant_type_ids": ["world_xxxx:TypeA", "world_xxxx:TypeB"],\n'
        '      "relation_pair": {"primary": "...", "bridge": "..."},\n'
        '      "difficulty": "easy|medium|hard",\n'
        '      "noise_tags": ["typo", "shorthand"],\n'
        '      "adversarial_tags": ["negation", "confusable_alias"],\n'
        '      "negative_type_ids": ["world_xxxx:TypeN"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def _call_openai(api_key: str, model: str, system: str, user: str, cfg: OpenAISeedConfig) -> dict[str, Any]:
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    return {"request": payload, "response": data, "parsed": parsed}


def _validate_items(parsed: dict[str, Any], valid_type_ids: set[str], canonical_names: set[str]) -> list[SeedItem]:
    batch = SeedBatch.model_validate(parsed)
    out: list[SeedItem] = []

    for item in batch.items:
        if item.difficulty not in DIFFICULTIES:
            continue
        if len(item.query.split()) < 4:
            continue
        qlow = item.query.lower()
        if any(x.lower() in qlow for x in canonical_names):
            continue
        if item.primary_type_id not in valid_type_ids:
            continue
        rel = [x for x in item.relevant_type_ids if x in valid_type_ids]
        if not rel:
            rel = [item.primary_type_id]
        if item.primary_type_id not in rel:
            rel.insert(0, item.primary_type_id)

        out.append(
            item.model_copy(
                update={
                    "relevant_type_ids": rel[:5],
                    "negative_type_ids": [x for x in item.negative_type_ids if x in valid_type_ids][:8],
                }
            )
        )
    return out


def _local_fallback_generate(world_id: str, domain: str, type_ids: list[str], aliases: dict[str, list[str]], n: int, seed: int) -> list[SeedItem]:
    rng = random.Random(seed)
    templates = [
        "I need to compare options for {alias} under budget limits",
        "Show me why my recent {alias} failed and what related records to inspect",
        "What data should I fetch to troubleshoot delayed {alias} updates",
        "Find entities linked to this {alias} with the highest risk",
        "I want results similar to this {alias} but not outdated ones",
        "How can I locate missing records around {alias} and dependencies",
    ]

    out = []
    for i in range(n):
        primary = type_ids[i % len(type_ids)]
        pool = [x for x in type_ids if x != primary]
        rel = rng.sample(pool, k=min(max(1, rng.randint(1, 3)), len(pool))) if pool else []
        rel = [primary] + rel
        bridge = rel[1] if len(rel) > 1 else primary
        alias = aliases.get(primary, [primary.split(":", 1)[1].lower()])[0]
        text = rng.choice(templates).format(alias=alias)
        out.append(
            SeedItem(
                query=f"[{domain}] {text}",
                primary_type_id=primary,
                relevant_type_ids=rel[:5],
                relation_pair={"primary": primary, "bridge": bridge},
                difficulty=rng.choice(DIFFICULTIES),
                noise_tags=[rng.choice(["", "typo", "shorthand", "locale"])],
                adversarial_tags=[rng.choice(["", "negation", "confusable_alias", "overloaded_term"])],
                negative_type_ids=rng.sample(pool, k=min(5, len(pool))) if pool else [],
            )
        )
    return out


def generate_openai_seed(
    out_dir: Path,
    cfg: OpenAISeedConfig,
    seed: int = 42,
    api_key: str | None = None,
    mock_responses_path: Path | None = None,
) -> tuple[list[QueryRecord], list[dict[str, Any]], list[CorpusRecord]]:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "openai")

    # Generate synthetic worlds first.
    worlds_meta, corpus_rows = generate_schema_worlds(
        out_dir=out_dir,
        cfg=WorldConfig(
            version=cfg.worlds_version,
            world_count=cfg.world_count,
            min_types=cfg.min_types_per_world,
            max_types=cfg.max_types_per_world,
            seed=seed,
        ),
    )

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    mock_payloads: list[dict[str, Any]] = []
    if mock_responses_path:
        mock_payloads = [json.loads(x) for x in mock_responses_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    raw_logs: list[dict[str, Any]] = []
    out_rows: list[QueryRecord] = []

    worlds_dir = out_dir / "worlds" / f"v{cfg.worlds_version}"

    for wi, world in enumerate(tqdm(worlds_meta, desc="OpenAI world query generation", unit="world")):
        world_id = world["world_id"]
        domain = world["domain"]
        split = world["split"]

        catalog_payload = read_json(worlds_dir / world_id / "type_catalog.json")
        catalog: dict[str, dict] = catalog_payload["types"]
        valid_type_ids = {f"{world_id}:{t}" for t in catalog}
        canonical = set(catalog.keys())

        parsed: dict[str, Any] | None = None
        raw: dict[str, Any] | None = None
        attempt = 0
        last_err: Exception | None = None

        while attempt < cfg.retries:
            try:
                if mock_payloads:
                    parsed = mock_payloads[wi % len(mock_payloads)]
                    raw = {"mock": True, "parsed": parsed}
                elif api_key:
                    raw = _call_openai(
                        api_key=api_key,
                        model=cfg.model,
                        system=_system_prompt(),
                        user=_user_prompt(world_id, domain, catalog, cfg),
                        cfg=cfg,
                    )
                    parsed = raw["parsed"]
                else:
                    # Offline fallback still honors new schema-world objective.
                    type_ids = sorted(valid_type_ids)
                    aliases = {f"{world_id}:{t}": (catalog[t].get("aliases") or [t.lower()]) for t in catalog}
                    items = _local_fallback_generate(world_id, domain, type_ids, aliases, cfg.items_per_world, seed + wi)
                    parsed = {"items": [x.model_dump() for x in items]}
                    raw = {"fallback": True, "parsed": parsed}
                break
            except (requests.RequestException, ValidationError, ValueError, KeyError) as exc:
                last_err = exc
                attempt += 1
                time.sleep(0.7 * attempt)

        if parsed is None:
            raise RuntimeError(f"Query generation failed for world {world_id}: {last_err}")

        valid_items = _validate_items(parsed, valid_type_ids=valid_type_ids, canonical_names=canonical)
        raw_logs.append(
            {
                "world_id": world_id,
                "split": split,
                "attempts": attempt + 1,
                "valid_items": len(valid_items),
                "raw": raw,
            }
        )

        for i, item in enumerate(valid_items):
            primary = item.primary_type_id
            bridge = (item.relation_pair or {}).get("bridge", primary)
            out_rows.append(
                QueryRecord(
                    query_id=f"openai-{world_id}-{i}",
                    query=item.query,
                    target_type_id=primary,
                    split=split,
                    source="openai-world-seed",
                    family_id=f"{world_id}-{i // 4}",
                    quality_score=0.92,
                    world_id=world_id,
                    domain=domain,
                    world_split=split,
                    primary_type_id=primary,
                    relevant_type_ids=item.relevant_type_ids,
                    relation_pair={"primary": primary, "bridge": bridge},
                    difficulty=item.difficulty,
                    noise_tags=[x for x in item.noise_tags if x],
                    adversarial_tags=[x for x in item.adversarial_tags if x],
                    negative_type_ids=item.negative_type_ids,
                    confuser_type_ids=item.negative_type_ids[:5],
                    rationale_tags=[domain, "world-seeded"],
                )
            )

    write_jsonl(out_dir / "openai" / "raw_seed_responses.jsonl", raw_logs)
    write_jsonl(out_dir / "openai" / "seed_pairs_v1.jsonl", [x.model_dump() for x in out_rows])
    return out_rows, raw_logs, corpus_rows
