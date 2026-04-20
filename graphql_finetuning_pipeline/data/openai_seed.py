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
from graphql_finetuning_pipeline.utils.io import ensure_dir, read_json, write_json, write_jsonl

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
    allow_local_fallback: bool = False
    min_real_row_fraction: float = 0.9
    # Real-schema inputs. When set, each file in ``real_schemas_dir`` is
    # ingested as its own world and mixed into the seed generation step.
    # Real worlds default to the "test" split unless overridden via
    # ``real_split_overrides`` — so that metrics reflect transfer to live
    # GraphQL schemas the model never saw during training.
    real_schemas_dir: str | None = None
    real_split_overrides: dict[str, str] | None = None
    # How many distinct phrasing styles to request per target coordinate.
    # See _user_prompt for the style definitions. 4 is a sensible default;
    # lower to 1 to regenerate the old one-phrasing-per-target behavior.
    phrasings_per_target: int = 4


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
        "You generate realistic natural-language search queries that map to a single GraphQL schema coordinate "
        "for a semantic-introspection field retriever. Users are engineers or LLM tool-callers; they describe "
        "the capability they need, NOT the canonical schema name. Your queries must be phrased the way a real "
        "developer would talk in Slack, in a ticket, or to an AI assistant. Strict JSON with top-level key 'items'."
    )


_PHRASING_STYLES = [
    ("plain", "A direct, one-clause question. Example: 'Which field gives me the creation time?'"),
    ("ellipsis", "Telegraphic phrasing with articles/verbs omitted. Example: 'field for creation time?' or 'creation time on this?'"),
    ("multi_clause", "Two-clause phrasing that establishes context before asking. Example: 'I already have the record loaded; which nested field tells me when it was created?'"),
    ("sibling_confuser", "A question that would naively lure a retriever into the wrong near-sibling (e.g. status vs statusHistory, author vs authorId, createdBy vs createdAt). Phrase it so the INTENDED coordinate is unambiguous to a human but the naive keyword match favors a sibling."),
]


def _user_prompt(world_id: str, domain: str, field_catalog: list[dict], item_count: int, phrasings_per_target: int) -> str:
    sample = field_catalog[: min(len(field_catalog), 80)]
    style_block = "\n".join(f"  - {name}: {desc}" for name, desc in _PHRASING_STYLES[:max(1, min(phrasings_per_target, len(_PHRASING_STYLES)))])
    unique_targets = max(1, item_count // max(1, phrasings_per_target))
    return (
        f"World id: {world_id}\n"
        f"Domain: {domain}\n"
        f"Generate exactly {item_count} queries total.\n"
        f"Pick ~{unique_targets} distinct target coordinates from the catalog and write {phrasings_per_target} different queries PER target, "
        f"one for EACH of these phrasing styles:\n{style_block}\n\n"
        "Rules:\n"
        "- A query must NOT contain the canonical owner type or field name (no leakage).\n"
        "- A query targets ONE best coordinate in positive_coordinate. If two coordinates truly both answer it, "
        "  include both in relevant_coordinates (path-to-root ambiguity).\n"
        "- Use natural human phrasing — conversational, terse, or business-y. Avoid generic boilerplate like 'which field tells me'.\n"
        "- difficulty: hard for sibling_confuser, medium for multi_clause/ellipsis, easy for plain.\n"
        "- intent: one of lookup | filtering | authorship | status | temporal | troubleshooting | comparison | list_vs_single | root_vs_nested.\n"
        "- For sibling_confuser style, set confuser_tags to include 'name_similarity' and list 2+ sibling coordinates in negative_coordinates "
        "  (e.g. if target is Order.status, include Order.statusHistory and Order.currentStatus when present).\n"
        "- For root_vs_nested/list_vs_single intents on object-typed fields, consider whether multiple path-equivalent coordinates "
        "  exist and use relevant_coordinates accordingly.\n"
        "- Prefer paraphrased domain aliases over the raw type name; the catalog provides aliases.\n"
        f"Field catalog JSON (use coordinate/owner_type/field_name/return_type/aliases to ground queries):\n"
        f"{json.dumps(sample, ensure_ascii=True)}\n"
        "Output schema (array of items, one per phrasing):\n"
        '{"items":[{"query":"...","positive_coordinate":"Type.field","relevant_coordinates":["Type.field"],'
        '"negative_coordinates":["Type.siblingField"],"difficulty":"hard","intent":"status",'
        '"confuser_tags":["name_similarity","same_owner"],"adversarial_tags":["sibling_confuser"],'
        '"noise_tags":[],"rationale_tags":["phrasing:sibling_confuser"]}]}'
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
    """Return a natural-language hint that does NOT echo the field or owner name.

    Earlier versions of this function produced hints like "status" or "servicerequest"
    that were literal canonical tokens, which then caused the query to be rejected
    by the validator. The hints below are deliberately paraphrased.
    """
    name = field["field_name"]
    mapping = {
        "author": "person who wrote it",
        "authorId": "writer reference",
        "createdBy": "person who created it",
        "updatedBy": "person who last changed it",
        "owner": "person responsible",
        "ownerId": "responsible-party reference",
        "createdAt": "creation time",
        "updatedAt": "last update time",
        "deletedAt": "deletion time",
        "publishedAt": "publication time",
        "status": "current state",
        "state": "current state",
        "email": "contact address",
        "phone": "contact number",
        "name": "human-readable label",
        "title": "headline text",
        "description": "long-form text body",
        "currency": "monetary unit",
        "priceCents": "price amount",
        "totalCents": "total amount",
        "amountCents": "amount value",
        "id": "unique reference",
    }
    if name in mapping:
        return mapping[name]
    # Synthesize a non-leaking hint from the return type and a few heuristics,
    # never from the literal field name.
    rt = (field.get("return_type") or "").replace("!", "").replace("[", "").replace("]", "").lower()
    if name.endswith("At") or rt in {"datetime", "date", "timestamp", "time"}:
        return "moment in time"
    if name.endswith("Id") or rt == "id":
        return "unique reference"
    if name.endswith("Cents") or rt in {"float", "int"}:
        return "numeric measurement"
    if rt == "boolean":
        return "yes/no flag"
    if rt == "string":
        return "short text value"
    return "related value"


# Map a field to the set of intents that semantically apply to it.
# The fallback only emits (field, intent) pairs that pass this test so
# it cannot produce rows like `Product.name` labeled as a temporal query.
_TEMPORAL_SUFFIXES = ("At", "Date", "Time")
_AUTHORSHIP_FIELDS = {"author", "authorId", "createdBy", "updatedBy", "owner", "ownerId", "writer", "editor", "assignee", "reporter"}
_STATUS_FIELDS = {"status", "state", "phase", "stage", "lifecycleStage", "orderStatus", "paymentStatus"}
_LOOKUP_SCALAR_FIELDS = {"email", "phone", "name", "title", "description", "slug", "url", "handle", "sku", "barcode", "currency", "locale"}
_FILTER_FIELDS = {"tag", "tags", "category", "categoryId", "type", "kind", "priority", "severity"}
_TROUBLESHOOTING_FIELDS = {"error", "errorMessage", "failureReason", "lastError", "errorCode"}
_COMPARISON_FIELDS = {"priceCents", "totalCents", "amountCents", "score", "rating", "rank", "count"}
_INTENT_MIN_LEN = 4


def _field_return_kind(field: dict) -> str:
    rt = (field.get("return_type") or "").replace("!", "").replace("[", "").replace("]", "")
    return rt


def _intents_for_field(field: dict) -> list[str]:
    """Return the intents that are *semantically valid* for this field.

    Never assign an intent that doesn't match the field — that is what
    produced the "Product.name → when was it created?" garbage labels in
    earlier runs.
    """
    name = field.get("field_name", "")
    rt = _field_return_kind(field)
    intents: list[str] = []

    if any(name.endswith(s) for s in _TEMPORAL_SUFFIXES) or rt in {"DateTime", "Date", "Timestamp", "Time"}:
        intents.append("temporal")
        return intents

    if name in _AUTHORSHIP_FIELDS or name.endswith("By"):
        intents.append("authorship")
    if name in _STATUS_FIELDS:
        intents.append("status")
    if name in _TROUBLESHOOTING_FIELDS:
        intents.append("troubleshooting")
    if name in _COMPARISON_FIELDS:
        intents.append("comparison")
    if name in _FILTER_FIELDS:
        intents.append("filtering")
    if name in _LOOKUP_SCALAR_FIELDS:
        intents.append("lookup")

    # root_vs_nested / list_vs_single are structural — only apply to object/list fields
    is_list = field.get("is_list") is True or "[" in (field.get("return_type") or "")
    if is_list:
        intents.append("list_vs_single")
    if rt and rt not in {"String", "Int", "Float", "Boolean", "ID", "DateTime", "Date", "Timestamp", "JSON"}:
        # object return type — reasonable target for root_vs_nested
        intents.append("root_vs_nested")

    # Fall back to plain lookup for unrecognized scalars. Never attach a
    # temporal/authorship/status intent to a field that has no evidence for it.
    if not intents:
        intents.append("lookup")
    return intents


def _local_fallback_generate(world_id: str, domain: str, field_catalog: list[dict], n: int, seed: int) -> list[SeedItem]:
    rng = random.Random(seed)
    out: list[SeedItem] = []
    usable = [f for f in field_catalog if f["field_name"] not in {"id"}]
    if not usable:
        usable = field_catalog
    if not usable:
        return out

    # Build (field, intent) pairs up front so we can shuffle across the
    # whole catalog rather than cycling field-major (which gives a very
    # lopsided intent distribution).
    pairs: list[tuple[dict, str]] = []
    for f in usable:
        for intent in _intents_for_field(f):
            pairs.append((f, intent))
    if not pairs:
        return out
    rng.shuffle(pairs)

    for i in range(n):
        target, intent = pairs[i % len(pairs)]
        # Prefer aliases that are paraphrased domain words, not variants of
        # the owner or field names themselves. Aliases in the world catalog
        # currently include both — so we filter to tokens that don't appear
        # anywhere in owner_type or field_name (case-insensitive substring).
        owner_low = target["owner_type"].lower()
        field_low = target["field_name"].lower()
        safe_aliases = [
            a for a in (target.get("aliases") or [])
            if a and a.lower() not in {owner_low, field_low}
            and a.lower() not in owner_low and a.lower() not in field_low
            and owner_low not in a.lower() and field_low not in a.lower()
        ]
        owner_label = safe_aliases[0] if safe_aliases else "record"
        hint = _hint_for_field(target)
        query = rng.choice(INTENT_TEMPLATES[intent]).format(owner_label=owner_label, hint=hint)
        # Defense in depth: skip if the final query still contains either
        # canonical token — rather than let downstream silently reject it.
        ql = query.lower()
        if owner_low in ql or field_low in ql:
            continue
        if len(query.split()) < _INTENT_MIN_LEN:
            # Skip rows that would be filtered downstream by ambiguity_filter anyway.
            continue
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
                query=query,
                positive_coordinate=target["coordinate"],
                relevant_coordinates=[target["coordinate"]],
                negative_coordinates=negatives,
                difficulty=rng.choice(DIFFICULTIES),
                intent=intent,
                confuser_tags=confuser_tags,
                adversarial_tags=adversarial,
                noise_tags=[],
                rationale_tags=[domain, "field-centric", "local-fallback"],
            )
        )
    return out


_TRIVIAL_CANONICAL = {"id", "name", "type", "value", "data", "item", "list", "count", "total", "date", "time", "user", "post", "order", "cart", "hotel", "flight", "trip", "product", "catalog"}


def _meaningful_canonical(terms: set[str]) -> set[str]:
    """Strip tokens that coincide with everyday English or domain labels.

    The original implementation rejected any query containing a canonical
    token — but canonical included trivial words like ``name``, ``id``,
    ``time``, so perfectly reasonable queries ("which field tells me when
    the item was created?") got dropped. Keep only multi-part coordinates
    and field names that aren't in the English-overlap allowlist.
    """
    out: set[str] = set()
    for t in terms:
        if "." in t and len(t) >= 5:
            out.add(t)  # "post.author"
            continue
        if len(t) >= 4 and t not in _TRIVIAL_CANONICAL:
            out.add(t)
    return out


def _validate_items(parsed: dict[str, Any], valid_coordinates: set[str], canonical_terms: set[str]) -> list[SeedItem]:
    if isinstance(parsed, dict) and "items" not in parsed and "query" in parsed:
        parsed = {"items": [parsed]}
    batch = SeedBatch.model_validate(parsed)
    filtered_canonical = _meaningful_canonical(canonical_terms)
    out: list[SeedItem] = []
    for item in batch.items:
        if item.difficulty not in DIFFICULTIES or item.intent not in INTENTS:
            continue
        if len(item.query.split()) < 4:
            continue
        qlow = item.query.lower()
        if any(term in qlow for term in filtered_canonical):
            continue
        if item.positive_coordinate not in valid_coordinates:
            continue
        relevant = [x for x in item.relevant_coordinates if x in valid_coordinates] or [item.positive_coordinate]
        negatives = [x for x in item.negative_coordinates if x in valid_coordinates and x != item.positive_coordinate]
        out.append(item.model_copy(update={"relevant_coordinates": relevant[:4], "negative_coordinates": negatives[:8]}))
    return out


def _run_generation_batch(*, world_id: str, domain: str, field_catalog: list[dict], req_n: int, batch_no: int, cfg: OpenAISeedConfig, api_key: str | None, mock_payloads: list[dict[str, Any]], world_index: int, seed: int, valid_coordinates: set[str], canonical: set[str], allow_local_fallback: bool) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    raw: dict[str, Any] | None = None
    attempt = 0
    last_err: Exception | None = None
    source = "openai"
    if mock_payloads:
        source = "mock"
    elif not api_key:
        source = "local-fallback"
    while attempt < cfg.retries:
        try:
            if mock_payloads:
                parsed = mock_payloads[(world_index + batch_no) % len(mock_payloads)]
                raw = {"mock": True, "parsed": parsed}
            elif api_key:
                raw = _call_openai(api_key=api_key, model=cfg.model, system=_system_prompt(), user=_user_prompt(world_id, domain, field_catalog, req_n, cfg.phrasings_per_target), cfg=cfg)
                parsed = raw["parsed"]
            else:
                if not allow_local_fallback:
                    raise RuntimeError(
                        "No OPENAI_API_KEY available and --allow-local-fallback is not set. "
                        "Export OPENAI_API_KEY, pass --api-key, or opt in to the lower-quality "
                        "local fallback with --allow-local-fallback."
                    )
                items = _local_fallback_generate(world_id, domain, field_catalog, req_n, seed + world_index + batch_no)
                parsed = {"items": [x.model_dump() for x in items]}
                raw = {"fallback": True, "parsed": parsed}
            break
        except (requests.RequestException, ValidationError, ValueError, KeyError) as exc:
            last_err = exc
            attempt += 1
            time.sleep(0.7 * attempt)
    if parsed is None:
        if not allow_local_fallback:
            return {
                "batch": batch_no,
                "requested": req_n,
                "attempts": attempt,
                "valid_items": 0,
                "taken_items": [],
                "raw": {"error": str(last_err) if last_err else "unknown", "source": "failed"},
                "source": "failed",
                "error": str(last_err) if last_err else "unknown",
            }
        items = _local_fallback_generate(world_id, domain, field_catalog, req_n, seed + world_index + 10_000 + batch_no)
        parsed = {"items": [x.model_dump() for x in items]}
        raw = {"fallback_after_error": str(last_err), "parsed": parsed}
        source = "local-fallback"
    try:
        valid_items = _validate_items(parsed, valid_coordinates=valid_coordinates, canonical_terms=canonical)
    except ValidationError as exc:
        if not allow_local_fallback:
            return {
                "batch": batch_no,
                "requested": req_n,
                "attempts": attempt + 1,
                "valid_items": 0,
                "taken_items": [],
                "raw": {"validation_error": str(exc), "parsed": parsed, "source": "failed"},
                "source": "failed",
                "error": str(exc),
            }
        items = _local_fallback_generate(world_id, domain, field_catalog, req_n, seed + world_index + 20_000 + batch_no)
        valid_items = _validate_items({"items": [x.model_dump() for x in items]}, valid_coordinates=valid_coordinates, canonical_terms=canonical)
        raw = {"fallback_after_validation_error": str(exc), "parsed": parsed}
        source = "local-fallback"
    return {"batch": batch_no, "requested": req_n, "attempts": attempt + 1, "valid_items": len(valid_items), "taken_items": valid_items[:req_n], "raw": raw, "source": source}


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

    if cfg.real_schemas_dir:
        from graphql_finetuning_pipeline.data.real_schemas import RealSchemaConfig, ingest_real_schemas

        real_worlds, real_corpus = ingest_real_schemas(
            sdl_dir=Path(cfg.real_schemas_dir),
            out_dir=out_dir,
            cfg=RealSchemaConfig(version=cfg.worlds_version, real_split_overrides=cfg.real_split_overrides),
        )
        # Extend the corpus and worlds manifest. Real worlds keep their
        # explicit split (default "test") so transfer is measured honestly.
        corpus_rows.extend(real_corpus)
        worlds_meta.extend(real_worlds)
        worlds_dir = out_dir / "worlds" / f"v{cfg.worlds_version}"
        write_json(
            worlds_dir / "manifest.json",
            {
                "version": cfg.worlds_version,
                "world_count": len(worlds_meta),
                "synthetic_worlds": sum(1 for w in worlds_meta if w.get("source") != "real-schema"),
                "real_worlds": sum(1 for w in worlds_meta if w.get("source") == "real-schema"),
                "worlds": worlds_meta,
            },
        )
        # Overwrite the merged corpus so downstream build-dataset sees real rows too.
        write_jsonl(out_dir / "corpus" / f"types_worlds_v{cfg.worlds_version}.jsonl", [c.model_dump() for c in corpus_rows])

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    mock_payloads: list[dict[str, Any]] = []
    if mock_responses_path:
        mock_payloads = [json.loads(x) for x in mock_responses_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    if not api_key and not mock_payloads and not cfg.allow_local_fallback:
        raise RuntimeError(
            "OpenAI seed generation requires OPENAI_API_KEY (or --api-key, or --mock-responses-path). "
            "To proceed with the lower-quality local fallback pass --allow-local-fallback explicitly. "
            "Silently falling back has previously produced mislabeled training data."
        )
    source_counts: dict[str, int] = {"openai": 0, "local-fallback": 0, "mock": 0, "failed": 0}

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
                        allow_local_fallback=cfg.allow_local_fallback,
                    )
                    for batch_no, req_n in chunk
                ]
                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                    except Exception as exc:
                        batch_logs.append({"batch": -1, "requested": 0, "attempts": 0, "valid_items": 0, "raw": {"thread_error": str(exc)}, "source": "failed"})
                        source_counts["failed"] += 1
                        inner.update(1)
                        continue
                    batch_source = res.get("source", "openai")
                    source_counts[batch_source] = source_counts.get(batch_source, 0) + len(res["taken_items"])
                    world_valid_items.extend([(item, batch_source) for item in res["taken_items"]])
                    batch_logs.append({"batch": res["batch"], "requested": res["requested"], "attempts": res["attempts"], "valid_items": res["valid_items"], "raw": res["raw"], "source": batch_source})
                    inner.update(1)
        inner.close()
        world_valid_items = world_valid_items[: cfg.items_per_world]

        raw_logs.append({"world_id": world_id, "split": split, "valid_items": len(world_valid_items), "batches": batch_logs})
        with partial_raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw_logs[-1], ensure_ascii=True) + "\n")

        for i, (item, batch_source) in enumerate(world_valid_items):
            source_tag = "openai-world-seed" if batch_source == "openai" else ("mock-seed" if batch_source == "mock" else "local-fallback")
            quality = 0.92 if batch_source == "openai" else (0.6 if batch_source == "mock" else 0.3)
            row = QueryRecord(
                query_id=f"{source_tag}-{world_id}-{i}",
                query=item.query,
                positive_coordinate=item.positive_coordinate,
                split=split,
                source=source_tag,
                family_id=f"{world_id}:{item.positive_coordinate}",
                quality_score=quality,
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

    total = sum(source_counts.values())
    real = source_counts["openai"] + source_counts["mock"]
    real_fraction = real / total if total else 0.0
    if api_key and not mock_payloads and real_fraction < cfg.min_real_row_fraction and not cfg.allow_local_fallback:
        raise RuntimeError(
            f"Only {real}/{total} rows ({real_fraction:.1%}) came from OpenAI; the rest fell back. "
            f"This is below the min_real_row_fraction={cfg.min_real_row_fraction:.0%} threshold. "
            "Check OpenAI connectivity and rate limits, or opt in to mixed data with --allow-local-fallback."
        )

    print(json.dumps({"source_counts": source_counts, "real_fraction": round(real_fraction, 4)}))
    write_jsonl(out_dir / "openai" / "raw_seed_responses.jsonl", raw_logs)
    write_jsonl(out_dir / "openai" / "seed_pairs_v1.jsonl", [x.model_dump() for x in out_rows])
    (out_dir / "openai" / "source_counts.json").write_text(json.dumps({"source_counts": source_counts, "real_fraction": real_fraction, "total_rows": total}, indent=2), encoding="utf-8")
    return out_rows, raw_logs, corpus_rows
