from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
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
    model: str = "gpt-5.4-mini-2026-03-17"
    temperature: float = 0.6
    max_tokens: int = 2400
    retries: int = 3
    timeout_seconds: int = 60
    items_per_world: int = 180
    request_batch_size: int = 30
    # Total in-flight OpenAI requests across ALL worlds. The flattened
    # request queue lets us saturate the OpenAI rate limit even when some
    # worlds have few batches; bump this to 16-32 for 60-world runs.
    max_concurrency: int = 16
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
    # Real-world SDLs have far more cross-type field-name ambiguity than
    # synthetic worlds (GHES: 120 ambiguous names; 8 owners on `.actor`).
    # Boosting their item budget increases the share of training rows that
    # genuinely require owner-type disambiguation.
    real_schema_boost: int = 3
    # Cap each prompt's competition set so GPT gets a clear contrast. GHES
    # `.actor` with 8 competitors is too noisy to write 5 disambiguating
    # queries for. 6 is a practical ceiling.
    competition_set_cap: int = 6
    # Adversarial hard-example mining against the base embedding model.
    # When enabled, every validated candidate is scored by the cold
    # base model against its competition set, then stratified so the
    # final seed is dominated by examples the base model gets wrong.
    adversarial_mining: bool = True
    adversarial_base_model: str = "Qwen/Qwen3-Embedding-0.6B"
    target_hard_fraction: float = 0.55
    target_medium_fraction: float = 0.30
    # Easy fraction is inferred as 1 - hard - medium. Keeping a small easy
    # floor prevents the fine-tune from drifting on basic retrieval.


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
        "You generate realistic natural-language queries that an end-user or LLM assistant would make "
        "when it needs a specific piece of information from an application. "
        "The user does NOT know the schema — they express a business need, a question about their data, "
        "or a task they want to accomplish. They never say 'field', 'schema', 'coordinate', or 'GraphQL'. "
        "Write the query as if the person is typing into a chat interface or search box. "
        "Strict JSON with top-level key 'items'."
    )


_PHRASING_STYLES = [
    ("plain", "A direct, natural question about the data. NO schema words. "
              "BAD: 'Which field gives me the creation time?' "
              "GOOD: 'When was this hotel added?' or 'How old is this listing?'"),
    ("ellipsis", "Terse, message-style phrasing. Like a Slack message. "
                 "BAD: 'field for creation time?' "
                 "GOOD: 'creation date?' or 'when added?' or 'hotel age?'"),
    ("multi_clause", "Conversational context + question. Reads like talking to a colleague or AI assistant. "
                     "BAD: 'I have the record loaded; which field tells me when it was created?' "
                     "GOOD: 'I'm looking at a booking and need to know when it was originally made — where does that live?'"),
    ("sibling_confuser", "A realistic user question whose answer is the TARGET coordinate, but a naive "
                         "keyword match would land on a near-sibling (e.g. createdAt vs updatedAt, "
                         "status vs statusHistory, author vs authorId). "
                         "The user intent must clearly point to the TARGET, not the sibling. "
                         "Do NOT mention sibling field names. "
                         "GOOD example for target=Order.updatedAt, sibling=Order.createdAt: "
                         "'When did this order last change?' (clearly updatedAt, not createdAt)"),
]


# Field names that are ambiguous-by-construction (every entity has one) but
# not useful training targets: no real user types "what's the id of the thing
# that tracks when the booking was created on my hotel" — they'd just ask for
# the creation time directly. We skip these at the ambiguity-map level so the
# prompt budget goes to names where owner disambiguation carries real intent.
_AMBIGUITY_SKIP = {"id", "createdat", "updatedat", "deletedat", "publishedat", "version", "type"}


def _ambiguity_map(field_catalog: list[dict]) -> dict[str, list[dict]]:
    """Group field_catalog by lowercase field_name; keep only names owned by ≥2 types.

    Each surviving bucket is a "competition set": fields that share a name but
    live on different owner types. These are exactly the targets whose
    correct retrieval requires the model to pick the owner type — the
    training signal we can't get from single-owner fields.

    Generic per-entity names (id, createdAt, …) are dropped: they're ambiguous
    across every world but carry no user intent that owner-disambiguation would
    serve. See `_AMBIGUITY_SKIP`.
    """
    from collections import defaultdict
    by_name: dict[str, list[dict]] = defaultdict(list)
    for f in field_catalog:
        name = f["field_name"].lower()
        if name in _AMBIGUITY_SKIP:
            continue
        by_name[name].append(f)
    return {name: fs for name, fs in by_name.items() if len({f["owner_type"] for f in fs}) >= 2}


def _render_competitor(field: dict, max_desc_chars: int = 140) -> str:
    """One-line representation of a competitor coordinate for the GPT prompt.

    Prefers owner description when available (real SDLs); falls back to
    the field description so synthetic worlds without owner descriptions
    still produce useful output.
    """
    owner_desc = (field.get("owner_description") or "").strip()
    field_desc = (field.get("description") or "").strip()
    desc = owner_desc or field_desc or ""
    if len(desc) > max_desc_chars:
        desc = desc[: max_desc_chars - 1].rstrip() + "…"
    rt = field.get("return_type") or "?"
    return f"  - {field['coordinate']}   ({rt})" + (f" — {desc}" if desc else "")


def _pick_competition_set(
    all_competitors: list[dict],
    target: dict,
    cap: int,
    rng: random.Random,
) -> list[dict]:
    """Choose up to ``cap`` competitors (including target) for one prompt.

    Real schemas like GHES put 8 types under `.actor`, which is too noisy
    for a single prompt. Sample competitors that are as descriptively
    distinct from the target as possible so GPT has meaningful contrast.
    """
    others = [f for f in all_competitors if f["coordinate"] != target["coordinate"]]
    if len(others) <= cap - 1:
        chosen = others
    else:
        # Prefer owners with descriptions, then a shuffled sample.
        with_desc = [f for f in others if (f.get("owner_description") or "").strip()]
        without = [f for f in others if not (f.get("owner_description") or "").strip()]
        rng.shuffle(with_desc)
        rng.shuffle(without)
        chosen = (with_desc + without)[: cap - 1]
    return [target] + chosen


def _user_prompt_competition(
    world_id: str,
    domain: str,
    world_source: str,
    target: dict,
    competition_set: list[dict],
    phrasings_per_target: int,
) -> str:
    """Prompt GPT to disambiguate ONE target from its owner-type competitors.

    The prompt's shape, not its wording, is what makes the queries hard:
    by listing every competing owner and its description, we force GPT to
    write queries whose context clues point to the target's owner — which
    is exactly the training signal the base model lacks.
    """
    style_block = "\n".join(
        f"  - {name}: {desc}"
        for name, desc in _PHRASING_STYLES[: max(1, min(phrasings_per_target, len(_PHRASING_STYLES)))]
    )
    competitors_block = "\n".join(_render_competitor(f) for f in competition_set if f["coordinate"] != target["coordinate"])
    owner_desc = (target.get("owner_description") or "").strip()
    path = target.get("path_to_root") or []
    path_line = " / ".join(path) if path else "Query.* → … → " + target["coordinate"]
    schema_label = "real-world SDL" if world_source == "real" else "synthetic domain schema"
    return (
        f"World: {world_id} ({domain}) — {schema_label}\n\n"
        "Retrieval-training query generation. The downstream system returns a GraphQL\n"
        "coordinate (Type.field) and then walks the schema graph to build the full\n"
        "path from Query.* down to the leaf. Picking the WRONG owner type produces\n"
        "a path that never reaches the user's answer, even when the field name matches.\n"
        "So the training signal we need is owner-type disambiguation, not field-name guessing.\n\n"
        f"TARGET coordinate: {target['coordinate']}\n"
        f"  return_type: {target.get('return_type', '?')}\n"
        f"  path_to_root: {path_line}\n"
        + (f"  owner description: {owner_desc}\n" if owner_desc else "")
        + (f"  field description: {(target.get('description') or '').strip()}\n" if target.get('description') else "")
        + "\n"
        "COMPETING coordinates (same field name, different owner — your query MUST\n"
        "clearly distinguish the target from ALL of these by using context clues\n"
        "about the owner type, not the field name):\n"
        f"{competitors_block if competitors_block else '  (none — this target has no cross-type siblings)'}\n\n"
        f"Write exactly {phrasings_per_target} queries, one per phrasing style below.\n"
        "Each query MUST:\n"
        "  - Use context clues pointing to the TARGET's owner type, not a competitor.\n"
        "  - NOT mention the field name, the owner type name, or any GraphQL term\n"
        "    (field, schema, coordinate, type, column, attribute, property, endpoint).\n"
        "  - Read like a real person typing into a chat or search box.\n"
        "  - Be answerable by this target coordinate only. If another coordinate in\n"
        "    the competition set also legitimately answers the query, list both in\n"
        "    relevant_coordinates (but keep positive_coordinate = the target).\n"
        "  - Populate negative_coordinates with 2–4 entries from the competition set.\n\n"
        f"Phrasing styles:\n{style_block}\n\n"
        "Difficulty/intent rules:\n"
        "  - difficulty: hard for sibling_confuser, medium for multi_clause/ellipsis, easy for plain.\n"
        "  - intent: one of lookup | filtering | authorship | status | temporal | troubleshooting | comparison | list_vs_single | root_vs_nested.\n"
        "  - For sibling_confuser style, set adversarial_tags=[\"sibling_confuser\"].\n\n"
        "Output strict JSON with top-level 'items':\n"
        '{"items":[{"query":"...","positive_coordinate":"' + target["coordinate"] + '",'
        '"relevant_coordinates":["' + target["coordinate"] + '"],'
        '"negative_coordinates":["<competitor coordinate>"],'
        '"difficulty":"hard","intent":"authorship",'
        '"confuser_tags":["same_field_name"],"adversarial_tags":["sibling_confuser"],'
        '"noise_tags":[],"rationale_tags":[]}]}'
    )


def _user_prompt_fallback(
    world_id: str,
    domain: str,
    field_catalog: list[dict],
    item_count: int,
    phrasings_per_target: int,
) -> str:
    """Fallback prompt for worlds with no cross-type ambiguity.

    Worlds this small don't produce interesting owner-disambiguation rows.
    We generate regular queries against the first ~N fields so the pipeline
    doesn't starve, but these rows will mostly land in the easy bucket
    after adversarial mining — which is fine, they serve as the
    don't-forget-basic-retrieval floor.
    """
    unique_targets = max(1, item_count // max(1, phrasings_per_target))
    targets = field_catalog[:unique_targets]
    style_block = "\n".join(
        f"  - {name}: {desc}"
        for name, desc in _PHRASING_STYLES[: max(1, min(phrasings_per_target, len(_PHRASING_STYLES)))]
    )
    return (
        f"World id: {world_id}\nDomain: {domain}\n"
        f"Generate exactly {item_count} queries total across these ~{unique_targets} targets:\n"
        f"{json.dumps([f['coordinate'] for f in targets], ensure_ascii=True)}\n\n"
        f"Write {phrasings_per_target} queries per target, one per style:\n{style_block}\n\n"
        "Hard rules:\n"
        "- NEVER use the words: field, schema, coordinate, GraphQL, type, column, attribute, property, endpoint.\n"
        "- NEVER mention the canonical owner type or field name verbatim.\n"
        "- Write what a real person types into a chat or search box.\n\n"
        "Output strict JSON with top-level 'items', one per query. Each item has:\n"
        "  query, positive_coordinate, relevant_coordinates, negative_coordinates,\n"
        "  difficulty (easy|medium|hard), intent (lookup|filtering|authorship|status|temporal|troubleshooting|comparison|list_vs_single|root_vs_nested),\n"
        "  confuser_tags, adversarial_tags, noise_tags, rationale_tags."
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


_TRIVIAL_CANONICAL = {
    "id", "name", "type", "value", "data", "item", "list", "count", "total",
    "date", "time", "user", "post", "order", "cart", "hotel", "flight", "trip",
    "product", "catalog", "unit", "text", "body", "record", "mode", "state",
    "status", "created", "updated", "amount", "price", "active", "label",
    "title", "notes", "start", "email", "phone", "image", "score", "error",
    "token", "slug", "field", "when", "last", "update", "create", "creation",
    "unique", "identifier", "display", "number", "model", "case", "times",
    "member",
}


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


# Words that signal the query is schema-navigation talk, not a real user question.
# A query containing any of these sounds like API documentation, not a user intent.
_SCHEMA_NAV_WORDS = {
    "field", "schema", "coordinate", "graphql", "column", "attribute",
    "property", "endpoint", "parameter", "which field", "what field",
    "same as the", "is the same as",
}


def _validate_items(parsed: dict[str, Any], valid_coordinates: set[str], canonical_terms: set[str]) -> list[SeedItem]:
    if isinstance(parsed, dict) and "items" not in parsed and "query" in parsed:
        parsed = {"items": [parsed]}
    batch = SeedBatch.model_validate(parsed)
    filtered_canonical = _meaningful_canonical(canonical_terms)
    out: list[SeedItem] = []
    for item in batch.items:
        if item.difficulty not in DIFFICULTIES or item.intent not in INTENTS:
            continue
        if len(item.query.split()) < 3:
            continue
        qlow = item.query.lower()
        # Reject schema-navigation language — real users don't say "which field"
        if any(word in qlow for word in _SCHEMA_NAV_WORDS):
            continue
        if any(term in qlow for term in filtered_canonical):
            continue
        if item.positive_coordinate not in valid_coordinates:
            continue
        relevant = [x for x in item.relevant_coordinates if x in valid_coordinates] or [item.positive_coordinate]
        negatives = [x for x in item.negative_coordinates if x in valid_coordinates and x != item.positive_coordinate]
        out.append(item.model_copy(update={"relevant_coordinates": relevant[:4], "negative_coordinates": negatives[:8]}))
    return out


def _run_competition_batch(
    *,
    world_id: str,
    domain: str,
    world_source: str,
    field_catalog: list[dict],
    target: dict,
    competition_set: list[dict],
    req_n: int,
    batch_no: int,
    cfg: OpenAISeedConfig,
    api_key: str | None,
    mock_payloads: list[dict[str, Any]],
    world_index: int,
    seed: int,
    valid_coordinates: set[str],
    canonical: set[str],
    allow_local_fallback: bool,
) -> dict[str, Any]:
    """One batch = one (target, competition_set) prompt.

    This replaces the legacy `_run_generation_batch`. Each batch now asks
    GPT for `req_n` queries ALL pointed at one target but generated with
    explicit knowledge of the competing owner types. That's the key
    restructuring: without a target-scoped prompt, GPT silently
    generates easy queries because it doesn't know what "hard" means.
    """
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
                if competition_set and len(competition_set) >= 2:
                    user_prompt = _user_prompt_competition(
                        world_id=world_id, domain=domain, world_source=world_source,
                        target=target, competition_set=competition_set,
                        phrasings_per_target=req_n,
                    )
                else:
                    # No competitors → fallback prompt; these rows form the
                    # easy floor the user asked to keep.
                    user_prompt = _user_prompt_fallback(world_id, domain, field_catalog, req_n, min(req_n, cfg.phrasings_per_target))
                raw = _call_openai(api_key=api_key, model=cfg.model, system=_system_prompt(), user=user_prompt, cfg=cfg)
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
                "competition_set": [f["coordinate"] for f in competition_set],
                "target": target["coordinate"] if target else None,
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
                "competition_set": [f["coordinate"] for f in competition_set],
                "target": target["coordinate"] if target else None,
                "raw": {"validation_error": str(exc), "parsed": parsed, "source": "failed"},
                "source": "failed",
                "error": str(exc),
            }
        items = _local_fallback_generate(world_id, domain, field_catalog, req_n, seed + world_index + 20_000 + batch_no)
        valid_items = _validate_items({"items": [x.model_dump() for x in items]}, valid_coordinates=valid_coordinates, canonical_terms=canonical)
        raw = {"fallback_after_validation_error": str(exc), "parsed": parsed}
        source = "local-fallback"
    return {
        "batch": batch_no,
        "requested": req_n,
        "attempts": attempt + 1,
        "valid_items": len(valid_items),
        "taken_items": valid_items[:req_n],
        "competition_set": [f["coordinate"] for f in competition_set],
        "target": target["coordinate"] if target else None,
        "raw": raw,
        "source": source,
    }


def generate_openai_seed(out_dir: Path, cfg: OpenAISeedConfig, seed: int = 42, api_key: str | None = None, mock_responses_path: Path | None = None) -> tuple[list[QueryRecord], list[dict[str, Any]], list[CorpusRecord]]:
    ensure_dir(out_dir)
    ensure_dir(out_dir / "openai")
    v = cfg.worlds_version
    partial_seed_path = out_dir / "openai" / f"seed_pairs_v{v}.partial.jsonl"
    partial_raw_path = out_dir / "openai" / f"raw_seed_responses_v{v}.partial.jsonl"
    # Batch-level checkpoint. One JSON line per completed batch, written
    # as soon as the future finishes — so Ctrl-C or crash mid-run preserves
    # OpenAI spend. On restart, already-done (world_id, batch_no) pairs
    # are loaded from here and skipped in phase 2.
    batches_checkpoint_path = out_dir / "openai" / f"batches_v{v}.jsonl"
    # Reset partial files only if no checkpoint exists (fresh run). Keep them
    # when resuming so the previous phase-3 output isn't truncated.
    resuming = batches_checkpoint_path.exists() and batches_checkpoint_path.stat().st_size > 0
    if not resuming:
        partial_seed_path.write_text("", encoding="utf-8")
        partial_raw_path.write_text("", encoding="utf-8")
        batches_checkpoint_path.write_text("", encoding="utf-8")

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

    bucket_totals: dict[str, int] = {"hard": 0, "medium": 0, "easy": 0}
    total_seed_written = 0
    world_rng = random.Random(seed)

    # ---- Phase 1: plan every batch up front, flatten across worlds ----
    # Per-world context loaded once so we don't re-parse field_catalog.json
    # 64 times in the critical path. Also lets Phase 2's fan-out draw
    # from a single pool of OpenAI calls, saturating the rate limit rather
    # than serialising world-by-world.
    @dataclass
    class _WorldCtx:
        wi: int
        world_id: str
        domain: str
        split: str
        world_source: str
        field_catalog: list[dict]
        valid_coordinates: set[str]
        canonical: set[str]

    world_ctx: dict[str, _WorldCtx] = {}
    all_batches: list[tuple[str, int, dict, list[dict], int]] = []  # (world_id, batch_no, target, comp_set, req_n)

    for wi, world in enumerate(worlds_meta):
        world_id = world["world_id"]
        domain = world["domain"]
        split = world["split"]
        world_source = "real" if world.get("source") == "real-schema" else "synthetic"
        field_catalog_payload = read_json(worlds_dir / world_id / "field_catalog.json")
        field_catalog = field_catalog_payload["fields"]
        valid_coordinates = {f["coordinate"] for f in field_catalog}
        canonical = (
            {f["coordinate"].lower() for f in field_catalog}
            | {f["owner_type"].lower() for f in field_catalog}
            | {f["field_name"].lower() for f in field_catalog}
        )
        world_ctx[world_id] = _WorldCtx(
            wi=wi, world_id=world_id, domain=domain, split=split, world_source=world_source,
            field_catalog=field_catalog, valid_coordinates=valid_coordinates, canonical=canonical,
        )

        ambiguity = _ambiguity_map(field_catalog)
        per_world_budget = cfg.items_per_world
        if world_source == "real":
            per_world_budget *= max(1, cfg.real_schema_boost)

        world_specs: list[tuple[int, dict, list[dict], int]] = []
        if ambiguity:
            names_sorted = sorted(ambiguity.keys())
            world_rng.shuffle(names_sorted)
            req_per_batch = max(1, cfg.phrasings_per_target)
            remaining = per_world_budget
            bn = 0
            for fname in names_sorted:
                if remaining <= 0:
                    break
                comp_all = ambiguity[fname]
                for target in comp_all:
                    if remaining <= 0:
                        break
                    comp_set = _pick_competition_set(comp_all, target, cfg.competition_set_cap, world_rng)
                    req_n = min(req_per_batch, remaining)
                    bn += 1
                    world_specs.append((bn, target, comp_set, req_n))
                    remaining -= req_n
        # Backfill with fallback prompts if ambiguity didn't fill the budget.
        filled = sum(x[3] for x in world_specs)
        deficit = per_world_budget - filled
        bn_start = (world_specs[-1][0] + 1) if world_specs else 1
        while deficit > 0:
            req_n = min(cfg.request_batch_size, deficit)
            world_specs.append((bn_start, {}, [], req_n))
            bn_start += 1
            deficit -= req_n

        for bn, target, comp_set, req_n in world_specs:
            all_batches.append((world_id, bn, target, comp_set, req_n))

    # ---- Phase 2: flat fan-out with checkpointing ----
    # One ThreadPool with max_concurrency workers across all worlds. An OpenAI
    # call is a blocking HTTP request, so raising concurrency above 4 tracks
    # linearly with rate limit headroom until saturation. 16 is a safe default
    # for tier-2 API keys; 32+ for tier-4.
    #
    # Each completed batch is streamed to batches_v{v}.jsonl immediately so
    # a crash/Ctrl-C doesn't waste the OpenAI spend. On restart we load that
    # file, rebuild per_world_results, and skip (world_id, batch_no) pairs
    # already present.
    per_world_results: dict[str, list[dict[str, Any]]] = {wid: [] for wid in world_ctx.keys()}

    done_keys: set[tuple[str, int]] = set()
    if resuming:
        with batches_checkpoint_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                wid = rec.get("_world_id")
                bn = rec.get("batch")
                if wid is None or bn is None:
                    continue
                # Strip helper metadata before rebuilding the batch result.
                res = {k: v for k, v in rec.items() if not k.startswith("_")}
                # Recover SeedItem objects from the serialized items.
                raw_items = res.get("taken_items") or []
                parsed_items: list[SeedItem] = []
                for it in raw_items:
                    try:
                        parsed_items.append(SeedItem.model_validate(it) if isinstance(it, dict) else it)
                    except Exception:
                        continue
                res["taken_items"] = parsed_items
                per_world_results.setdefault(wid, []).append(res)
                done_keys.add((wid, int(bn)))
                # Restore source counts so the final real_fraction check is accurate.
                bsrc = res.get("source", "openai")
                source_counts[bsrc] = source_counts.get(bsrc, 0) + len(parsed_items)

    pending = [b for b in all_batches if (b[0], b[1]) not in done_keys]
    total_batches = len(all_batches)
    skipped = len(all_batches) - len(pending)
    progress = tqdm(
        total=total_batches,
        initial=skipped,
        desc=f"OpenAI batches ({cfg.max_concurrency} in flight{', resume' if skipped else ''})",
        unit="batch",
    )

    # Append-mode checkpoint writer. Flushed after every batch to survive Ctrl-C.
    import threading
    ckpt_lock = threading.Lock()

    def _checkpoint(wid: str, res: dict[str, Any]) -> None:
        # Normalise SeedItems to dicts so the line is JSON-serialisable.
        serial = dict(res)
        items = serial.get("taken_items") or []
        serial["taken_items"] = [it.model_dump() if hasattr(it, "model_dump") else it for it in items]
        serial["_world_id"] = wid
        with ckpt_lock:
            with batches_checkpoint_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(serial, ensure_ascii=True, default=str) + "\n")
                f.flush()

    with ThreadPoolExecutor(max_workers=max(1, cfg.max_concurrency)) as executor:
        futures = [
            executor.submit(
                _run_competition_batch,
                world_id=world_id,
                domain=world_ctx[world_id].domain,
                world_source=world_ctx[world_id].world_source,
                field_catalog=world_ctx[world_id].field_catalog,
                target=target,
                competition_set=comp_set,
                req_n=req_n,
                batch_no=batch_no,
                cfg=cfg,
                api_key=api_key,
                mock_payloads=mock_payloads,
                world_index=world_ctx[world_id].wi,
                seed=seed,
                valid_coordinates=world_ctx[world_id].valid_coordinates,
                canonical=world_ctx[world_id].canonical,
                allow_local_fallback=cfg.allow_local_fallback,
            )
            for (world_id, batch_no, target, comp_set, req_n) in pending
        ]
        fut_to_wid = {fut: pending[i][0] for i, fut in enumerate(futures)}
        try:
            for fut in as_completed(futures):
                wid = fut_to_wid[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    err_res = {
                        "batch": -1, "requested": 0, "attempts": 0, "valid_items": 0,
                        "taken_items": [], "competition_set": [], "target": None,
                        "raw": {"thread_error": str(exc)}, "source": "failed",
                    }
                    per_world_results[wid].append(err_res)
                    _checkpoint(wid, err_res)
                    source_counts["failed"] += 1
                    progress.update(1)
                    continue
                batch_source = res.get("source", "openai")
                source_counts[batch_source] = source_counts.get(batch_source, 0) + len(res["taken_items"])
                per_world_results[wid].append(res)
                _checkpoint(wid, res)
                progress.set_postfix(world=wid, buckets=bucket_totals)
                progress.update(1)
        except KeyboardInterrupt:
            progress.close()
            raise
    progress.close()

    # ---- Phase 3: per-world mining + write-out ----
    outer = tqdm(worlds_meta, desc="Mining + write", unit="world")
    for wi, world in enumerate(outer):
        world_id = world["world_id"]
        ctx = world_ctx[world_id]
        domain, split, world_source = ctx.domain, ctx.split, ctx.world_source
        field_catalog = ctx.field_catalog
        valid_coordinates = ctx.valid_coordinates

        world_results = per_world_results.get(world_id, [])
        world_valid_items: list[tuple[SeedItem, str, list[str]]] = []
        batch_logs: list[dict[str, Any]] = []
        for res in world_results:
            batch_source = res.get("source", "openai")
            for item in res.get("taken_items", []):
                world_valid_items.append((item, batch_source, res.get("competition_set") or []))
            batch_logs.append({
                "batch": res["batch"], "requested": res["requested"], "attempts": res.get("attempts", 0),
                "valid_items": res.get("valid_items", 0), "raw": res.get("raw"), "source": batch_source,
                "competition_set": res.get("competition_set"),
                "target": res.get("target"),
            })

        # Adversarial mining: score each item by the cold base model against
        # its own competition set, then stratify into hard/medium/easy so the
        # final seed is dominated by rows the base model fails on.
        scored_payload: list[tuple[SeedItem, str, dict[str, Any]]] = []  # (item, source, score_dict)
        if cfg.adversarial_mining and world_valid_items:
            # Group by competition-set so encoding cost amortizes.
            from collections import defaultdict
            from graphql_finetuning_pipeline.data.adversarial_mining import score_candidates, stratify, bucket_counts

            # Resolve coordinate → catalog entry for mining's competitor text.
            coord_to_field = {f["coordinate"]: f for f in field_catalog}

            grouped: dict[tuple[str, ...], list[SeedItem]] = defaultdict(list)
            group_source: dict[tuple[str, ...], str] = {}
            for item, src, comp_coords in world_valid_items:
                key = tuple(comp_coords) if comp_coords else ()
                grouped[key].append(item)
                group_source.setdefault(key, src)

            all_scored = []
            for key, group_items in grouped.items():
                comp_fields = [coord_to_field[c] for c in key if c in coord_to_field]
                sc = score_candidates(group_items, comp_fields, base_model=cfg.adversarial_base_model)
                for s in sc:
                    scored_payload.append((s.item, group_source[key], {
                        "bucket": s.bucket,
                        "base_top1_correct": s.base_top1_correct,
                        "base_target_rank": s.base_target_rank,
                        "base_margin": round(s.base_margin, 4),
                        "base_top1_wrong_coord": s.base_top1_wrong_coord,
                    }))
                all_scored.extend(sc)

            # Stratify at the world level so each world contributes a mix.
            kept = stratify(
                all_scored,
                target_hard_fraction=cfg.target_hard_fraction,
                target_medium_fraction=cfg.target_medium_fraction,
            )
            kept_ids = {id(s.item) for s in kept}
            scored_payload = [(item, src, meta) for (item, src, meta) in scored_payload if id(item) in kept_ids]
            counts = bucket_counts(kept)
            for b, n in counts.items():
                bucket_totals[b] = bucket_totals.get(b, 0) + n
        else:
            scored_payload = [(item, src, {"bucket": "unmined", "base_top1_correct": None,
                                           "base_target_rank": -1, "base_margin": 0.0,
                                           "base_top1_wrong_coord": None})
                              for item, src, _ in world_valid_items]

        raw_logs.append({"world_id": world_id, "split": split, "valid_items": len(scored_payload), "batches": batch_logs})
        with partial_raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw_logs[-1], ensure_ascii=True) + "\n")

        for i, (item, batch_source, meta) in enumerate(scored_payload):
            source_tag = "openai-world-seed" if batch_source == "openai" else ("mock-seed" if batch_source == "mock" else "local-fallback")
            quality = 0.92 if batch_source == "openai" else (0.6 if batch_source == "mock" else 0.3)
            # Stamp mining-derived tags for downstream filters + negative mining.
            rationale = list(item.rationale_tags) + [f"base_{meta['bucket']}", f"margin={meta['base_margin']}"]
            adversarial = list(item.adversarial_tags)
            if meta.get("base_top1_wrong_coord"):
                adversarial.append(f"base_model_confuses_with:{meta['base_top1_wrong_coord']}")
            # Promote the base-model-top-1-wrong coord into the first slot of
            # negative_coordinates so downstream MNRL mines the right negative.
            neg = list(item.negative_coordinates)
            wrong = meta.get("base_top1_wrong_coord")
            if wrong and wrong in valid_coordinates and wrong != item.positive_coordinate:
                neg = [wrong] + [n for n in neg if n != wrong]

            row = QueryRecord(
                query_id=f"{source_tag}-{world_id}-{i}",
                query=item.query,
                positive_coordinate=item.positive_coordinate,
                split=split,
                source=source_tag,
                family_id=f"{world_id}:{item.positive_coordinate}",
                quality_score=quality,
                negative_coordinates=neg[:8],
                relevant_coordinates=item.relevant_coordinates or [item.positive_coordinate],
                owner_type=item.positive_coordinate.split(".", 1)[0],
                field_name=item.positive_coordinate.split(".", 1)[1],
                world_id=world_id,
                domain=domain,
                world_split=split,
                difficulty=item.difficulty,
                intent=item.intent,
                confuser_tags=item.confuser_tags,
                adversarial_tags=adversarial,
                noise_tags=item.noise_tags,
                rationale_tags=rationale,
                confuser_coordinates=neg[:5],
            )
            out_rows.append(row)
            with partial_seed_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row.model_dump(), ensure_ascii=True) + "\n")
        total_seed_written += len(scored_payload)
        outer.set_postfix(world=world_id, generated=total_seed_written, buckets=bucket_totals)
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

    print(json.dumps({"source_counts": source_counts, "real_fraction": round(real_fraction, 4), "bucket_totals": bucket_totals}))
    write_jsonl(out_dir / "openai" / f"raw_seed_responses_v{v}.jsonl", raw_logs)
    write_jsonl(out_dir / "openai" / f"seed_pairs_v{v}.jsonl", [x.model_dump() for x in out_rows])
    # Checkpoint served its purpose — remove so a future rerun of the same
    # version starts fresh rather than resuming against stale state.
    try:
        batches_checkpoint_path.unlink()
    except FileNotFoundError:
        pass
    (out_dir / "openai" / "source_counts.json").write_text(
        json.dumps({
            "source_counts": source_counts,
            "real_fraction": real_fraction,
            "total_rows": total,
            "bucket_totals": bucket_totals,
        }, indent=2),
        encoding="utf-8",
    )
    return out_rows, raw_logs, corpus_rows
