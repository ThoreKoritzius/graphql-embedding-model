from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import random
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
INTENTS = [
    "lookup",
    "filtering",
    "authorship",
    "status",
    "temporal",
    "troubleshooting",
    "comparison",
    "list_vs_single",
    "root_vs_nested",
]


class SeedItem(BaseModel):
    query: str
    positive_coordinate: str
    relevant_coordinates: list[str] = Field(default_factory=list)
    negative_coordinates: list[str] = Field(default_factory=list)
    difficulty: str
    intent: str
    confuser_tags: list[str] = Field(default_factory=list)
    adversarial_tags: list[str] = Field(default_factory=list)
    noise_tags: list[str] = Field(default_factory=list)
    rationale_tags: list[str] = Field(default_factory=list)


class SeedBatch(BaseModel):
    items: list[SeedItem]


@dataclass
class OpenAISeedConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.6
    max_tokens: int = 2400
    retries: int = 3
    timeout_seconds: int = 60
    items_per_world: int = 180
    request_batch_size: int = 30
    max_concurrency: int = 4
    worlds_version: int = 1
    world_count: int = 60
    min_types_per_world: int = 20
    max_types_per_world: int = 36


INTENT_TEMPLATES = {
    "lookup": [
        "Where can I read the {hint}?",
        "Which field tells me the {hint}?",
    ],
    "filtering": [
        "Which field should I use to filter by {hint}?",
        "I need a field for narrowing results by {hint}",
    ],
    "authorship": [
        "Who is responsible for this {owner_label}?",
        "Who wrote the {owner_label}?",
    ],
    "status": [
        "What field tells me whether this {owner_label} is active or pending?",
        "How do I read the current state of a {owner_label}?",
    ],
    "temporal": [
        "When was this {owner_label} last updated?",
        "Which field tells me when the {owner_label} was created?",
    ],
    "troubleshooting": [
        "I need the field that explains why this {owner_label} failed",
        "What field should I inspect when this {owner_label} looks wrong?",
    ],
    "comparison": [
        "Which field helps compare one {owner_label} with another?",
        "What field would I inspect to compare {owner_label} records?",
    ],
    "list_vs_single": [
        "Which field on the {owner_label} record gives me the specific {hint}?",
        "I already have the {owner_label}; what field returns the {hint}?",
    ],
    "root_vs_nested": [
        "I already fetched the {owner_label}; which nested field gives me the {hint}?",
        "Not the root query, the field on {owner_label} that gives the {hint}",
    ],
}


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError("No valid JSON object found in model output")


def _system_prompt() -> str:
    return (
        "You generate realistic natural-language search queries for GraphQL semantic introspection field retrieval. "
        "Users do not know canonical schema names. Return strict JSON with top-level key 'items'."
    )


def _user_prompt(world_id: str, domain: str, field_catalog: list[dict], item_count: int) -> str:
    sample = field_catalog[: min(len(field_catalog), 80)]
    return (
        f"World id: {world_id}\n"
        f"Domain: {domain}\n"
        f"Generate {item_count} search queries.\n"
        "Constraints:\n"
        "- Queries must not include canonical owner or field names.\n"
        "- Target a single best schema coordinate.\n"
        "- Use realistic capability phrasing, not keyword bags.\n"
        "- Add hard confusers and adversarial wording by default.\n"
        "- difficulty must be easy|medium|hard.\n"
        f"Field catalog JSON:\n{json.dumps(sample, ensure_ascii=True)}\n"
        "Output schema:\n"
        '{"items":[{"query":"...","positive_coordinate":"Type.field","relevant_coordinates":["Type.field"],"negative_coordinates":["Type.otherField"],"difficulty":"hard","intent":"lookup","confuser_tags":["same_owner"],"adversarial_tags":["root_vs_nested"],"noise_tags":[],"rationale_tags":["field-centric"]}]}'
    )


def _call_openai(api_key: str, model: str, system: str, user: str, cfg: OpenAISeedConfig) -> dict[str, Any]:
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=cfg.timeout_seconds)
    resp.raise_for_status()
    data = resp.json()
    return {"request": payload, "response": data, "parsed": _extract_json(data["choices"][0]["message"]["content"])}


def _confusers(target: dict, field_catalog: list[dict], rng: random.Random) -> tuple[list[str], list[str]]:
    same_owner = [f["coordinate"] for f in field_catalog if f["owner_type"] == target["owner_type"] and f["coordinate"] != target["coordinate"]]
    same_field = [f["coordinate"] for f in field_catalog if f["field_name"] == target["field_name"] and f["owner_type"] != target["owner_type"]]
    same_return = [f["coordinate"] for f in field_catalog if f["return_type"] == target["return_type"] and f["coordinate"] != target["coordinate"]]
    negs = []
    tags = []
    for tag, pool in [("same_owner", same_owner), ("same_field_name", same_field), ("same_return_type", same_return)]:
        if pool:
            negs.append(rng.choice(pool))
            tags.append(tag)
    if not negs:
        pool = [f["coordinate"] for f in field_catalog if f["coordinate"] != target["coordinate"]]
        negs = rng.sample(pool, k=min(3, len(pool))) if pool else []
        tags.append("semantic")
    return list(dict.fromkeys(negs))[:6], tags


def _hint_for_field(field: dict) -> str:
    name = field["field_name"]
    mapping = {
        "author": "person who authored it",
        "authorId": "author identifier",
        "createdAt": "creation time",
        "updatedAt": "last update time",
        "status": "current state",
        "email": "email address",
        "phone": "phone number",
        "name": "displayed name",
        "currency": "currency",
        "priceCents": "price",
        "totalCents": "total amount",
        "id": "identifier",
    }
    return mapping.get(name, name.replace("Id", " identifier").replace("Cents", " amount").lower())


def _local_fallback_generate(world_id: str, domain: str, field_catalog: list[dict], n: int, seed: int) -> list[SeedItem]:
    rng = random.Random(seed)
    out: list[SeedItem] = []
    usable = [f for f in field_catalog if f["field_name"] not in {"id"}]
    if not usable:
        usable = field_catalog
    for i in range(n):
        target = usable[i % len(usable)]
        owner_label = (target.get("aliases") or [target["owner_type"].lower()])[0]
        hint = _hint_for_field(target)
        intent = rng.choice(INTENTS)
        query = rng.choice(INTENT_TEMPLATES[intent]).format(owner_label=owner_label, hint=hint)
        negatives, confuser_tags = _confusers(target, field_catalog, rng)
        adversarial = []
        if any(tag == "same_owner" for tag in confuser_tags):
            adversarial.append("owner_vs_field")
        if any(tag == "same_field_name" for tag in confuser_tags):
            adversarial.append("alias_collision")
        if intent == "root_vs_nested":
            adversarial.append("root_vs_nested")
        out.append(
            SeedItem(
                query=f"[{domain}] {query}",
                positive_coordinate=target["coordinate"],
                relevant_coordinates=[target["coordinate"]],
                negative_coordinates=negatives,
                difficulty=rng.choice(DIFFICULTIES),
                intent=intent,
                confuser_tags=confuser_tags,
                adversarial_tags=adversarial,
                noise_tags=[],
                rationale_tags=[domain, "field-centric"],
            )
        )
    return out


def _validate_items(parsed: dict[str, Any], valid_coordinates: set[str], canonical_terms: set[str]) -> list[SeedItem]:
    if isinstance(parsed, dict) and "items" not in parsed and "query" in parsed:
        parsed = {"items": [parsed]}
    batch = SeedBatch.model_validate(parsed)
    out: list[SeedItem] = []
    for item in batch.items:
        if item.difficulty not in DIFFICULTIES or item.intent not in INTENTS:
            continue
        if len(item.query.split()) < 4:
            continue
        qlow = item.query.lower()
        if any(term in qlow for term in canonical_terms):
            continue
        if item.positive_coordinate not in valid_coordinates:
            continue
        relevant = [x for x in item.relevant_coordinates if x in valid_coordinates] or [item.positive_coordinate]
        negatives = [x for x in item.negative_coordinates if x in valid_coordinates and x != item.positive_coordinate]
        out.append(item.model_copy(update={"relevant_coordinates": relevant[:4], "negative_coordinates": negatives[:8]}))
    return out


def _run_generation_batch(*, world_id: str, domain: str, field_catalog: list[dict], req_n: int, batch_no: int, cfg: OpenAISeedConfig, api_key: str | None, mock_payloads: list[dict[str, Any]], world_index: int, seed: int, valid_coordinates: set[str], canonical: set[str]) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    attempt = 0
    last_err: Exception | None = None
    while attempt < cfg.retries:
        try:
            if mock_payloads:
                parsed = mock_payloads[(world_index + batch_no) % len(mock_payloads)]
                raw = {"mock": True, "parsed": parsed}
            elif api_key:
                raw = _call_openai(api_key=api_key, model=cfg.model, system=_system_prompt(), user=_user_prompt(world_id, domain, field_catalog, req_n), cfg=cfg)
                parsed = raw["parsed"]
            else:
                items = _local_fallback_generate(world_id, domain, field_catalog, req_n, seed + world_index + batch_no)
                parsed = {"items": [x.model_dump() for x in items]}
                raw = {"fallback": True, "parsed": parsed}
            break
        except (requests.RequestException, ValidationError, ValueError, KeyError) as exc:
            last_err = exc
            attempt += 1
            time.sleep(0.7 * attempt)
    if parsed is None:
        items = _local_fallback_generate(world_id, domain, field_catalog, req_n, seed + world_index + 10_000 + batch_no)
        parsed = {"items": [x.model_dump() for x in items]}
        raw = {"fallback_after_error": str(last_err), "parsed": parsed}
    try:
        valid_items = _validate_items(parsed, valid_coordinates=valid_coordinates, canonical_terms=canonical)
    except ValidationError as exc:
        items = _local_fallback_generate(world_id, domain, field_catalog, req_n, seed + world_index + 20_000 + batch_no)
        valid_items = _validate_items({"items": [x.model_dump() for x in items]}, valid_coordinates=valid_coordinates, canonical_terms=canonical)
        raw = {"fallback_after_validation_error": str(exc), "parsed": parsed}
    return {"batch": batch_no, "requested": req_n, "attempts": attempt + 1, "valid_items": len(valid_items), "taken_items": valid_items[:req_n], "raw": raw}


def generate_openai_seed(out_dir: Path, cfg: OpenAISeedConfig, seed: int = 42, api_key: str | None = None, mock_responses_path: Path | None = None) -> tuple[list[QueryRecord], list[dict[str, Any]], list[CorpusRecord]]:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "openai")
    partial_seed_path = out_dir / "openai" / "seed_pairs_v1.partial.jsonl"
    partial_raw_path = out_dir / "openai" / "raw_seed_responses.partial.jsonl"
    partial_seed_path.write_text("", encoding="utf-8")
    partial_raw_path.write_text("", encoding="utf-8")

    worlds_meta, corpus_rows = generate_schema_worlds(
        out_dir=out_dir,
        cfg=WorldConfig(version=cfg.worlds_version, world_count=cfg.world_count, min_types=cfg.min_types_per_world, max_types=cfg.max_types_per_world, seed=seed),
    )

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    mock_payloads: list[dict[str, Any]] = []
    if mock_responses_path:
        mock_payloads = [json.loads(x) for x in mock_responses_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    raw_logs: list[dict[str, Any]] = []
    out_rows: list[QueryRecord] = []
    worlds_dir = out_dir / "worlds" / f"v{cfg.worlds_version}"

    outer = tqdm(worlds_meta, desc="OpenAI world query generation", unit="world")
    total_seed_written = 0
    for wi, world in enumerate(outer):
        world_id = world["world_id"]
        domain = world["domain"]
        split = world["split"]
        field_catalog_payload = read_json(worlds_dir / world_id / "field_catalog.json")
        field_catalog: list[dict] = field_catalog_payload["fields"]
        valid_coordinates = {f["coordinate"] for f in field_catalog}
        canonical = {f["coordinate"].lower() for f in field_catalog} | {f["owner_type"].lower() for f in field_catalog} | {f["field_name"].lower() for f in field_catalog}

        total_batches = max(math.ceil(cfg.items_per_world / max(cfg.request_batch_size, 1)), 1)
        batch_specs: list[tuple[int, int]] = []
        remaining_target = cfg.items_per_world
        for b in range(1, total_batches + 1):
            req_n = min(cfg.request_batch_size, remaining_target)
            batch_specs.append((b, req_n))
            remaining_target -= req_n

        world_valid_items: list[SeedItem] = []
        batch_logs: list[dict[str, Any]] = []
        inner = tqdm(total=total_batches, desc=f"{world_id} batches", unit="batch", leave=False)
        for i in range(0, len(batch_specs), max(1, cfg.max_concurrency)):
            chunk = batch_specs[i : i + max(1, cfg.max_concurrency)]
            with ThreadPoolExecutor(max_workers=max(1, cfg.max_concurrency)) as executor:
                futures = [
                    executor.submit(
                        _run_generation_batch,
                        world_id=world_id,
                        domain=domain,
                        field_catalog=field_catalog,
                        req_n=req_n,
                        batch_no=batch_no,
                        cfg=cfg,
                        api_key=api_key,
                        mock_payloads=mock_payloads,
                        world_index=wi,
                        seed=seed,
                        valid_coordinates=valid_coordinates,
                        canonical=canonical,
                    )
                    for batch_no, req_n in chunk
                ]
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        batch_logs.append({"batch": -1, "requested": 0, "attempts": 0, "valid_items": 0, "raw": {"thread_error": str(exc)}})
                        inner.update(1)
                        continue
                    world_valid_items.extend(res["taken_items"])
                    batch_logs.append({"batch": res["batch"], "requested": res["requested"], "attempts": res["attempts"], "valid_items": res["valid_items"], "raw": res["raw"]})
                    inner.update(1)
        inner.close()
        world_valid_items = world_valid_items[: cfg.items_per_world]

        raw_logs.append({"world_id": world_id, "split": split, "valid_items": len(world_valid_items), "batches": batch_logs})
        with partial_raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw_logs[-1], ensure_ascii=True) + "\n")

        for i, item in enumerate(world_valid_items):
            row = QueryRecord(
                query_id=f"openai-{world_id}-{i}",
                query=item.query,
                positive_coordinate=item.positive_coordinate,
                split=split,
                source="openai-world-seed",
                family_id=f"{world_id}:{item.positive_coordinate}",
                quality_score=0.92,
                negative_coordinates=item.negative_coordinates,
                relevant_coordinates=item.relevant_coordinates or [item.positive_coordinate],
                owner_type=item.positive_coordinate.split(".", 1)[0],
                field_name=item.positive_coordinate.split(".", 1)[1],
                world_id=world_id,
                domain=domain,
                world_split=split,
                difficulty=item.difficulty,
                intent=item.intent,
                confuser_tags=item.confuser_tags,
                adversarial_tags=item.adversarial_tags,
                noise_tags=item.noise_tags,
                rationale_tags=item.rationale_tags,
                confuser_coordinates=item.negative_coordinates[:5],
            )
            out_rows.append(row)
            with partial_seed_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row.model_dump(), ensure_ascii=True) + "\n")
        total_seed_written += len(world_valid_items)
        outer.set_postfix(world=world_id, generated=total_seed_written)
    outer.close()

    write_jsonl(out_dir / "openai" / "raw_seed_responses.jsonl", raw_logs)
    write_jsonl(out_dir / "openai" / "seed_pairs_v1.jsonl", [x.model_dump() for x in out_rows])
    return out_rows, raw_logs, corpus_rows
