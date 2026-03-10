from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_jsonl

INTENTS = ["lookup", "capability", "filtering", "aggregation", "debugging", "comparative"]
DIFFICULTIES = ["easy", "medium", "hard"]


class SeedItem(BaseModel):
    query: str
    target_type_id: str
    intent: str
    difficulty: str
    confuser_type_ids: list[str] = Field(default_factory=list)
    rationale_tags: list[str] = Field(default_factory=list)


class SeedBatch(BaseModel):
    items: list[SeedItem]


@dataclass
class OpenAISeedConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1600
    retries: int = 3
    timeout_seconds: int = 60
    items_per_type: int = 24
    max_confusers: int = 5


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
        "You generate high-quality retrieval training examples for mapping user GraphQL questions to a target type. "
        "Return strict JSON only with top-level key 'items'."
    )


def _user_prompt(target: CorpusRecord, corpus: list[CorpusRecord], cfg: OpenAISeedConfig) -> str:
    other = [c.type_id for c in corpus if c.type_id != target.type_id][: cfg.max_confusers * 3]
    return (
        f"Target type id: {target.type_id}\n"
        f"Target summary: {target.short_text}\n"
        f"Target full doc: {target.full_text}\n"
        f"Possible confusers: {other}\n"
        f"Generate {cfg.items_per_type} items with balanced intents {INTENTS} and difficulties {DIFFICULTIES}.\n"
        "Schema:\n"
        "{\n"
        '  "items": [\n'
        "    {\n"
        '      "query": "...",\n'
        '      "target_type_id": "...",\n'
        '      "intent": "lookup|capability|filtering|aggregation|debugging|comparative",\n'
        '      "difficulty": "easy|medium|hard",\n'
        '      "confuser_type_ids": ["..."],\n'
        '      "rationale_tags": ["semantic", "ambiguous", "business"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules: query must not contain the exact target type id string."
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


def _validate_items(parsed: dict[str, Any], target_type_id: str) -> list[SeedItem]:
    batch = SeedBatch.model_validate(parsed)
    valid: list[SeedItem] = []
    for item in batch.items:
        if item.target_type_id != target_type_id:
            continue
        if item.intent not in INTENTS or item.difficulty not in DIFFICULTIES:
            continue
        if len(item.query.split()) < 4:
            continue
        valid.append(item)
    return valid


def generate_openai_seed(
    corpus: list[CorpusRecord],
    out_dir: Path,
    cfg: OpenAISeedConfig,
    seed: int = 42,
    api_key: str | None = None,
    mock_responses_path: Path | None = None,
) -> tuple[list[QueryRecord], list[dict[str, Any]]]:
    del seed
    ensure_dir(out_dir)
    ensure_dir(out_dir / "openai")

    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key and not mock_responses_path:
        raise RuntimeError("OPENAI_API_KEY is required unless --mock-responses-path is provided")

    mock_payloads: list[dict[str, Any]] = []
    if mock_responses_path:
        mock_payloads = [json.loads(x) for x in mock_responses_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    raw_logs: list[dict[str, Any]] = []
    out_rows: list[QueryRecord] = []

    for idx, target in enumerate(tqdm(corpus, desc="OpenAI seed generation", unit="type")):
        system = _system_prompt()
        user = _user_prompt(target, corpus, cfg)

        attempt = 0
        last_err: Exception | None = None
        parsed: dict[str, Any] | None = None
        raw: dict[str, Any] | None = None

        while attempt < cfg.retries:
            try:
                if mock_payloads:
                    parsed = mock_payloads[idx % len(mock_payloads)]
                    raw = {"mock": True, "parsed": parsed}
                else:
                    raw = _call_openai(api_key=api_key or "", model=cfg.model, system=system, user=user, cfg=cfg)
                    parsed = raw["parsed"]
                break
            except (requests.RequestException, ValidationError, ValueError, KeyError) as exc:
                last_err = exc
                attempt += 1
                time.sleep(0.8 * attempt)

        if parsed is None:
            raise RuntimeError(f"OpenAI generation failed for type {target.type_id}: {last_err}")

        valid_items = _validate_items(parsed, target_type_id=target.type_id)
        raw_logs.append(
            {
                "type_id": target.type_id,
                "attempts": attempt + 1,
                "valid_items": len(valid_items),
                "raw": raw,
            }
        )

        for i, item in enumerate(valid_items):
            out_rows.append(
                QueryRecord(
                    query_id=f"openai-{target.type_id}-{i}",
                    query=item.query,
                    target_type_id=item.target_type_id,
                    split="train",
                    source="openai-seed",
                    family_id=f"openai-{target.type_id}-{item.intent}",
                    quality_score=0.9,
                    intent=item.intent,
                    difficulty=item.difficulty,
                    confuser_type_ids=item.confuser_type_ids,
                    rationale_tags=item.rationale_tags,
                )
            )

    write_jsonl(out_dir / "openai" / "raw_seed_responses.jsonl", raw_logs)
    write_jsonl(out_dir / "openai" / "seed_pairs_v1.jsonl", [x.model_dump() for x in out_rows])
    return out_rows, raw_logs
