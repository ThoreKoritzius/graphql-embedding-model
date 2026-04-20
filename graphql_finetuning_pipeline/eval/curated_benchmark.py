"""Merge a hand-written realism benchmark into the dataset's benchmark dir.

Synthetic benchmarks can be gamed by the same generator that built them.
The release gate that actually matters is a small hand-curated benchmark
against your *real* target schema, written by a human who knows the
domain. This module ingests such a file, validates it against the corpus,
and writes it as ``curated_realism_eval.jsonl`` next to the other
benchmark suites.

Input format (JSONL, one row per line) - minimal fields:

    {"query": "who wrote the post?", "positive_coordinate": "Post.author",
     "relevant_coordinates": ["Post.author"],
     "negative_coordinates": ["User.name", "Post.authorId"],
     "intent": "authorship", "difficulty": "medium"}

Optional fields match QueryRecord (``confuser_tags``, ``rationale_tags``,
``world_id``, ``domain``).
"""
from __future__ import annotations

import json
from pathlib import Path

from graphql_finetuning_pipeline.data.models import CorpusRecord, QueryRecord
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_jsonl


def merge_curated_benchmark(
    curated_jsonl: Path,
    corpus: list[CorpusRecord],
    out_dir: Path,
    *,
    name: str = "curated_realism_eval",
) -> dict:
    if not curated_jsonl.exists():
        raise FileNotFoundError(f"Curated benchmark file not found: {curated_jsonl}")

    valid_coords = {c.coordinate for c in corpus}
    coord_to_corpus = {c.coordinate: c for c in corpus}
    rows: list[QueryRecord] = []
    rejected: list[dict] = []

    for i, raw in enumerate(curated_jsonl.read_text(encoding="utf-8").splitlines()):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            rejected.append({"line": i + 1, "reason": f"invalid json: {exc}"})
            continue
        pos = payload.get("positive_coordinate")
        if not pos or pos not in valid_coords:
            rejected.append({"line": i + 1, "reason": f"positive_coordinate not in corpus: {pos}"})
            continue
        relevant = [c for c in payload.get("relevant_coordinates", [pos]) if c in valid_coords] or [pos]
        negatives = [c for c in payload.get("negative_coordinates", []) if c in valid_coords and c not in relevant]
        corpus_row = coord_to_corpus[pos]
        row = QueryRecord(
            query_id=payload.get("query_id") or f"curated-realism-{i}",
            query=payload["query"],
            positive_coordinate=pos,
            split="test",
            source="curated-realism",
            family_id=payload.get("family_id") or f"curated:{pos}",
            quality_score=1.0,
            negative_coordinates=negatives,
            relevant_coordinates=relevant,
            owner_type=corpus_row.owner_type,
            field_name=corpus_row.field_name,
            world_id=payload.get("world_id") or corpus_row.metadata.get("world_id"),
            domain=payload.get("domain") or corpus_row.metadata.get("domain"),
            world_split="test",
            difficulty=payload.get("difficulty", "medium"),
            intent=payload.get("intent", "lookup"),
            confuser_tags=payload.get("confuser_tags", []),
            adversarial_tags=payload.get("adversarial_tags", []),
            noise_tags=payload.get("noise_tags", []),
            rationale_tags=payload.get("rationale_tags", ["curated", "release-gate"]),
            confuser_coordinates=negatives[:5],
        )
        rows.append(row)

    bench_dir = out_dir / "benchmarks"
    ensure_dir(bench_dir)
    out_path = bench_dir / f"{name}.jsonl"
    write_jsonl(out_path, [r.model_dump() for r in rows])

    return {
        "out_path": str(out_path),
        "rows_kept": len(rows),
        "rows_rejected": len(rejected),
        "rejection_reasons": rejected[:20],
    }
