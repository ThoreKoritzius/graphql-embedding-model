"""Re-stratify an existing v{N} seed using a different bucket mix.

Loads the raw batch responses from a prior `generate-openai-seed` run, re-runs
the competition-set validator + adversarial mining, and writes a new
`seed_pairs_v{N}.jsonl` with the requested hard/medium/easy fractions.

Why: running generate-openai-seed again re-spends the OpenAI quota. The raw
log already contains every validated GPT candidate — this lets us dial the
stratification without re-paying.

Usage:
  python tools/remine_seed.py \
    --raw artifacts/openai/raw_seed_responses_v6.jsonl \
    --out artifacts/openai/seed_pairs_v6_loose.jsonl \
    --target-hard-fraction 0.40 \
    --target-medium-fraction 0.35 \
    --version 6 \
    --out-dir artifacts
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from graphql_finetuning_pipeline.data.adversarial_mining import (
    bucket_counts,
    score_candidates,
    stratify,
)
from graphql_finetuning_pipeline.data.openai_seed import (
    SeedItem,
    _extract_json,
    _validate_items,
)
from graphql_finetuning_pipeline.data.models import QueryRecord
from graphql_finetuning_pipeline.utils.io import read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="Path to raw_seed_responses_v{N}.jsonl from a prior run.")
    ap.add_argument("--out", required=True, help="Where to write the new seed_pairs_*.jsonl.")
    ap.add_argument("--out-dir", required=True, help="Artifacts root (needs worlds/v{N}/<wid>/field_catalog.json).")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--target-hard-fraction", type=float, default=0.40)
    ap.add_argument("--target-medium-fraction", type=float, default=0.35)
    ap.add_argument("--adversarial-base-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--skip-mining", action="store_true", help="Skip re-mining and keep all validated items. Fastest path to a large seed.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    worlds_dir = out_dir / "worlds" / f"v{args.version}"

    # Load the worlds manifest for splits + world metadata.
    manifest = read_json(worlds_dir / "manifest.json")
    world_splits = {w["world_id"]: w["split"] for w in manifest["worlds"]}
    world_domains = {w["world_id"]: w["domain"] for w in manifest["worlds"]}

    # Re-validate every batch's raw GPT response against its field_catalog.
    # Items collected per-world so mining has the right competition sets.
    all_rows: list[QueryRecord] = []
    bucket_totals: dict[str, int] = {"hard": 0, "medium": 0, "easy": 0}

    with open(args.raw, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            world_id = rec["world_id"]
            split = world_splits.get(world_id, rec.get("split", "train"))
            domain = world_domains.get(world_id, "unknown")

            field_catalog = read_json(worlds_dir / world_id / "field_catalog.json")["fields"]
            valid_coordinates = {f["coordinate"] for f in field_catalog}
            canonical = (
                {f["coordinate"].lower() for f in field_catalog}
                | {f["owner_type"].lower() for f in field_catalog}
                | {f["field_name"].lower() for f in field_catalog}
            )
            coord_to_field = {f["coordinate"]: f for f in field_catalog}

            # Collect (item, competition_set) across all batches for this world.
            world_items: list[tuple[SeedItem, list[str]]] = []
            for b in rec.get("batches", []):
                raw = b.get("raw") or {}
                parsed = raw.get("parsed")
                if not parsed:
                    resp = raw.get("response") or {}
                    try:
                        parsed = _extract_json(resp["choices"][0]["message"]["content"])
                    except Exception:
                        continue
                try:
                    items = _validate_items(parsed, valid_coordinates=valid_coordinates, canonical_terms=canonical)
                except Exception:
                    continue
                comp_coords = b.get("competition_set") or []
                for it in items:
                    world_items.append((it, comp_coords))

            if not world_items:
                continue

            # Mine or pass through.
            if args.skip_mining:
                kept_items = [(it, {"bucket": "unmined", "base_top1_wrong_coord": None, "base_margin": 0.0}) for it, _ in world_items]
            else:
                grouped: dict[tuple[str, ...], list[SeedItem]] = defaultdict(list)
                for it, coords in world_items:
                    grouped[tuple(coords)].append(it)

                all_scored = []
                score_meta: dict[int, dict] = {}
                for key, group in grouped.items():
                    comp_fields = [coord_to_field[c] for c in key if c in coord_to_field]
                    sc = score_candidates(group, comp_fields, base_model=args.adversarial_base_model)
                    all_scored.extend(sc)
                    for s in sc:
                        score_meta[id(s.item)] = {
                            "bucket": s.bucket,
                            "base_top1_wrong_coord": s.base_top1_wrong_coord,
                            "base_margin": round(s.base_margin, 4),
                        }

                kept = stratify(
                    all_scored,
                    target_hard_fraction=args.target_hard_fraction,
                    target_medium_fraction=args.target_medium_fraction,
                )
                counts = bucket_counts(kept)
                for b, n in counts.items():
                    bucket_totals[b] = bucket_totals.get(b, 0) + n
                kept_items = [(s.item, score_meta[id(s.item)]) for s in kept]

            for i, (item, meta) in enumerate(kept_items):
                rationale = list(item.rationale_tags) + [f"base_{meta['bucket']}", f"margin={meta['base_margin']}"]
                adversarial = list(item.adversarial_tags)
                if meta.get("base_top1_wrong_coord"):
                    adversarial.append(f"base_model_confuses_with:{meta['base_top1_wrong_coord']}")
                neg = list(item.negative_coordinates)
                wrong = meta.get("base_top1_wrong_coord")
                if wrong and wrong in valid_coordinates and wrong != item.positive_coordinate:
                    neg = [wrong] + [n for n in neg if n != wrong]
                row = QueryRecord(
                    query_id=f"openai-world-seed-{world_id}-{i}",
                    query=item.query,
                    positive_coordinate=item.positive_coordinate,
                    split=split,
                    source="openai-world-seed",
                    family_id=f"{world_id}:{item.positive_coordinate}",
                    quality_score=0.92,
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
                all_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r.model_dump(), ensure_ascii=True) + "\n")

    print(json.dumps({
        "rows_written": len(all_rows),
        "bucket_totals": bucket_totals,
        "out": str(out_path),
    }, indent=2))


if __name__ == "__main__":
    main()
