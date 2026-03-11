from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from graphql_finetuning_pipeline.data.models import CorpusRecord
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_json, write_jsonl

DOMAINS: dict[str, list[str]] = {
    "hotels": ["Hotel", "Room", "Guest", "Booking", "Amenity", "Location", "Review", "RatePlan", "Stay"],
    "shopping": ["Product", "Catalog", "Cart", "Order", "Customer", "Payment", "Shipment", "Inventory", "Brand"],
    "cars": ["Vehicle", "Model", "Dealer", "Listing", "Owner", "ServiceRecord", "Inspection", "PriceQuote", "Loan"],
    "travel": ["Trip", "Flight", "Airline", "Passenger", "Itinerary", "Seat", "Ticket", "Airport", "Baggage"],
    "fintech": ["Account", "Transaction", "Merchant", "Card", "Invoice", "Payout", "Balance", "Transfer", "RiskSignal"],
    "healthcare": ["Patient", "Provider", "Appointment", "Prescription", "Diagnosis", "Claim", "Coverage", "Facility", "LabResult"],
    "support": ["Ticket", "Issue", "Agent", "Queue", "SLA", "Conversation", "Customer", "Resolution", "Attachment"],
}

ALIASES = {
    "Hotel": ["property", "lodging"],
    "Room": ["unit", "suite"],
    "Booking": ["reservation", "stay booking"],
    "Order": ["purchase", "checkout"],
    "Product": ["item", "sku"],
    "Vehicle": ["car", "auto"],
    "Transaction": ["payment event", "movement"],
    "Patient": ["member", "person record"],
    "Ticket": ["case", "request"],
}


@dataclass
class WorldConfig:
    version: int = 1
    world_count: int = 50
    min_types: int = 20
    max_types: int = 36
    min_relations: int = 2
    max_relations: int = 4
    seed: int = 42


def _sdl_for_world(type_catalog: dict[str, dict]) -> str:
    lines = [
        '"Synthetic world query root"',
        "type Query {",
    ]
    for t in type_catalog:
        if t == "Query":
            continue
        lines.append(f"  {t[0].lower() + t[1:]}(id: ID!): {t}")
        lines.append(f"  {t[0].lower() + t[1:]}s(limit: Int, offset: Int): [{t}!]!")
    lines.append("}")
    lines.append("")

    for t, meta in type_catalog.items():
        if t == "Query":
            continue
        lines.append(f"type {t} implements Node {{")
        lines.append("  id: ID!")
        lines.append("  name: String")
        for rel in meta["neighbors"]:
            lines.append(f"  {rel[0].lower()+rel[1:]}: {rel}")
            lines.append(f"  {rel[0].lower()+rel[1:]}s(limit: Int): [{rel}!]!")
        lines.append("}")
        lines.append("")

    lines.append("interface Node {")
    lines.append("  id: ID!")
    lines.append("}")
    return "\n".join(lines)


def _full_text(type_name: str, neighbors: list[str], aliases: list[str], domain: str) -> str:
    return (
        f"Synthetic GraphQL type {type_name} in domain {domain}. "
        f"User-facing aliases: {', '.join(aliases) or 'none'}. "
        f"Related types: {', '.join(neighbors) or 'none'}. "
        "Fields include id, name, and relation edges to neighboring domain entities."
    )


def _type_catalog(domain: str, types: list[str], rng: random.Random, cfg: WorldConfig) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for t in types:
        n_rel = rng.randint(cfg.min_relations, min(cfg.max_relations, max(len(types) - 1, 1)))
        pool = [x for x in types if x != t]
        neighbors = rng.sample(pool, k=min(n_rel, len(pool))) if pool else []
        aliases = ALIASES.get(t, [t.lower(), f"{t.lower()} record"])[:3]
        catalog[t] = {
            "type_name": t,
            "domain": domain,
            "aliases": aliases,
            "semantic_tags": [domain, "entity"],
            "neighbors": neighbors,
        }
    return catalog


def generate_schema_worlds(out_dir: Path, cfg: WorldConfig) -> tuple[list[dict], list[CorpusRecord]]:
    rng = random.Random(cfg.seed)
    worlds_dir = out_dir / "worlds" / f"v{cfg.version}"
    ensure_dir(worlds_dir)

    world_rows: list[dict] = []
    corpus_rows: list[CorpusRecord] = []

    domains = list(DOMAINS.keys())
    for wi in range(cfg.world_count):
        world_id = f"world_{wi:04d}"
        domain = domains[wi % len(domains)]
        base_types = DOMAINS[domain]

        count = rng.randint(cfg.min_types, cfg.max_types)
        # Expand with synthetic variants to reach target type count.
        all_types = list(base_types)
        idx = 0
        while len(all_types) < count:
            all_types.append(f"{base_types[idx % len(base_types)]}Variant{len(all_types)}")
            idx += 1

        catalog = _type_catalog(domain, all_types, rng, cfg)

        world_dir = worlds_dir / world_id
        ensure_dir(world_dir)
        write_json(world_dir / "type_catalog.json", {"world_id": world_id, "domain": domain, "types": catalog})

        sdl = _sdl_for_world(catalog)
        (world_dir / "schema.graphql").write_text(sdl, encoding="utf-8")

        for t, meta in catalog.items():
            type_id = f"{world_id}:{t}"
            corpus_rows.append(
                CorpusRecord(
                    type_id=type_id,
                    type_name=t,
                    short_text=f"{t} entity in {domain}",
                    full_text=_full_text(t, meta["neighbors"], meta["aliases"], domain),
                    keywords_text=" | ".join(sorted(set([t.lower(), domain] + [a.lower() for a in meta["aliases"]]))),
                    metadata={
                        "world_id": world_id,
                        "domain": domain,
                        "neighbors": [f"{world_id}:{n}" for n in meta["neighbors"]],
                        "aliases": meta["aliases"],
                    },
                )
            )

        world_rows.append(
            {
                "world_id": world_id,
                "domain": domain,
                "type_count": len(catalog),
                "path": str(world_dir.resolve()),
            }
        )

    # World-level split to prevent leakage.
    total = len(world_rows)
    n_train = int(total * 0.8)
    n_val = int(total * 0.1)
    for i, wr in enumerate(world_rows):
        wr["split"] = "test"
        if i < n_train:
            wr["split"] = "train"
        elif i < n_train + n_val:
            wr["split"] = "val"

    write_json(worlds_dir / "manifest.json", {"version": cfg.version, "world_count": len(world_rows), "worlds": world_rows})
    write_jsonl(worlds_dir / "worlds.jsonl", world_rows)
    write_jsonl(out_dir / "corpus" / f"types_worlds_v{cfg.version}.jsonl", [c.model_dump() for c in corpus_rows])
    return world_rows, corpus_rows
