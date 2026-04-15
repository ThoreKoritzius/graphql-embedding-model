from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from graphql_finetuning_pipeline.data.models import CorpusRecord
from graphql_finetuning_pipeline.utils.io import ensure_dir, write_json, write_jsonl

DOMAINS: dict[str, list[str]] = {
    "hotels": [
        "Hotel",
        "Room",
        "Guest",
        "Booking",
        "Reservation",
        "RatePlan",
        "Stay",
        "Amenity",
        "PropertyLocation",
        "HotelReview",
        "LoyaltyAccount",
        "CheckInRequest",
        "HousekeepingTask",
        "CancellationPolicy",
        "BookingChannel",
        "AvailabilityWindow",
        "PaymentAuthorization",
        "RefundRequest",
        "PromoOffer",
        "OccupancyForecast",
        "TravelCompanion",
        "RoomUpgradeOffer",
        "IncidentReport",
        "Invoice",
        "TaxBreakdown",
        "GuestPreference",
        "AccessPass",
        "MaintenanceTicket",
        "ServiceRequest",
        "SeasonalPricingRule",
        "NearbyAttraction",
    ],
    "shopping": [
        "Product",
        "Catalog",
        "Cart",
        "Order",
        "Customer",
        "Payment",
        "Shipment",
        "InventoryItem",
        "Brand",
        "Collection",
        "Wishlist",
        "DiscountRule",
        "CouponRedemption",
        "ReturnRequest",
        "Refund",
        "ProductReview",
        "Merchant",
        "StoreLocation",
        "FulfillmentCenter",
        "PriceAdjustment",
        "SubscriptionPlan",
        "RecommendationSet",
        "AbandonedCartEvent",
        "TaxRate",
        "GiftCard",
        "AddressBookEntry",
        "LoyaltyReward",
        "FraudSignal",
        "OrderStatusEvent",
        "PromotionBanner",
    ],
    "cars": [
        "Vehicle",
        "VehicleModel",
        "Dealer",
        "VehicleListing",
        "OwnerProfile",
        "ServiceRecord",
        "InspectionReport",
        "PriceQuote",
        "AutoLoan",
        "InsurancePolicy",
        "MaintenanceSchedule",
        "RecallNotice",
        "WarrantyPlan",
        "MileageSnapshot",
        "RegistrationRecord",
        "VehicleHistoryEvent",
        "LeaseContract",
        "TestDriveBooking",
        "TradeInOffer",
        "FuelEfficiencyRating",
        "ChargingStation",
        "ChargingSession",
        "PartsInventory",
        "RepairEstimate",
        "RoadsideCase",
        "OwnershipTransfer",
        "AccidentReport",
        "TelematicsSignal",
        "ComplianceCheck",
        "DealerPromotion",
    ],
    "travel": [
        "Trip",
        "Flight",
        "Airline",
        "Passenger",
        "Itinerary",
        "SeatAssignment",
        "Ticket",
        "Airport",
        "BaggageItem",
        "BoardingPass",
        "CheckInSession",
        "FareRule",
        "LoyaltyTier",
        "HotelReservation",
        "CarRentalBooking",
        "TravelInsurance",
        "VisaRequirement",
        "DelayNotification",
        "ConnectionOption",
        "TerminalGate",
        "RouteSegment",
        "AncillaryService",
        "RefundEligibility",
        "PriorityService",
        "UpgradeOffer",
        "CompensationClaim",
        "WeatherAdvisory",
        "ScheduleChange",
        "TransitPass",
        "PartnerProgram",
    ],
    "fintech": [
        "Account",
        "Transaction",
        "Merchant",
        "Card",
        "Invoice",
        "Payout",
        "BalanceSnapshot",
        "Transfer",
        "RiskSignal",
        "ChargebackCase",
        "SettlementBatch",
        "KycProfile",
        "FraudAlert",
        "CardAuthorization",
        "DisputeEvidence",
        "PaymentIntent",
        "BankConnection",
        "SubscriptionBilling",
        "CreditLine",
        "RepaymentPlan",
        "PortfolioPosition",
        "MarketOrder",
        "LimitOrder",
        "ComplianceEvent",
        "TaxDocument",
        "StatementPeriod",
        "FeeSchedule",
        "CashFlowForecast",
        "FundingSource",
        "IdentityVerification",
    ],
    "healthcare": [
        "Patient",
        "Provider",
        "Appointment",
        "Prescription",
        "Diagnosis",
        "Claim",
        "CoveragePlan",
        "Facility",
        "LabResult",
        "CarePlan",
        "AllergyRecord",
        "MedicationOrder",
        "ProcedureRequest",
        "Referral",
        "Encounter",
        "VitalSign",
        "ImagingStudy",
        "PriorAuthorization",
        "BillingStatement",
        "CareTeamMember",
        "ImmunizationRecord",
        "DischargeSummary",
        "PatientConsent",
        "TelehealthSession",
        "ClinicalNote",
        "InsuranceEligibility",
        "SymptomReport",
        "ConditionTimeline",
        "ProviderSchedule",
        "CoverageException",
    ],
    "support": [
        "Ticket",
        "Issue",
        "Agent",
        "Queue",
        "SlaPolicy",
        "Conversation",
        "Customer",
        "Resolution",
        "Attachment",
        "Escalation",
        "PriorityRule",
        "Tag",
        "MessageEvent",
        "KnowledgeArticle",
        "RootCause",
        "StatusTransition",
        "AutomationRule",
        "SatisfactionSurvey",
        "ContactChannel",
        "WorklogEntry",
        "OwnershipChange",
        "Incident",
        "ServiceComponent",
        "OutageWindow",
        "Postmortem",
        "Notification",
        "Runbook",
        "QueueBacklogSnapshot",
        "FollowUpTask",
        "EscalationPath",
    ],
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


def _field_description(type_name: str, field_name: str, return_type: str) -> str:
    semantics = {
        "author": "Author or owner responsible for this record.",
        "status": "Current lifecycle state.",
        "createdAt": "Timestamp when the record was created.",
        "updatedAt": "Timestamp when the record last changed.",
        "email": "Email address used for communication or lookup.",
        "phone": "Phone number used for communication.",
        "displayName": "Human-readable name shown in UI.",
        "name": "Human-readable canonical name.",
        "currency": "ISO currency code.",
    }
    default = f"{field_name} on {type_name} returning {return_type}."
    return semantics.get(field_name, default)


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
        for field in meta.get("fields", []):
            lines.append(f"  {field['name']}: {field['type']}")
        for rel in meta["neighbors"]:
            lines.append(f"  {rel[0].lower()+rel[1:]}: {rel}")
            lines.append(f"  {rel[0].lower()+rel[1:]}s(limit: Int): [{rel}!]!")
        lines.append("}")
        lines.append("")

    lines.append("interface Node {")
    lines.append("  id: ID!")
    lines.append("}")
    return "\n".join(lines)


def _path_to_root(type_name: str, field_name: str) -> list[str]:
    return [f"{type_name}.{field_name}", f"Query.{type_name[0].lower() + type_name[1:]}"]


def _type_fields(type_name: str, domain: str) -> list[dict[str, str]]:
    lower = type_name.lower()
    base = [{"name": "id", "type": "ID!"}, {"name": "createdAt", "type": "String"}, {"name": "updatedAt", "type": "String"}]
    if any(x in lower for x in ["hotel", "property", "facility"]):
        base += [{"name": "name", "type": "String!"}, {"name": "city", "type": "String"}, {"name": "rating", "type": "Float"}]
    elif any(x in lower for x in ["room", "seat", "ticket"]):
        base += [{"name": "label", "type": "String!"}, {"name": "status", "type": "String"}, {"name": "priceCents", "type": "Int"}]
    elif any(x in lower for x in ["order", "reservation", "booking", "invoice", "claim"]):
        base += [{"name": "status", "type": "String!"}, {"name": "totalCents", "type": "Int"}, {"name": "currency", "type": "String"}]
    elif any(x in lower for x in ["customer", "guest", "patient", "owner", "agent", "provider", "passenger"]):
        base += [{"name": "displayName", "type": "String"}, {"name": "email", "type": "String"}, {"name": "phone", "type": "String"}]
    elif any(x in lower for x in ["payment", "transaction", "transfer", "payout", "refund"]):
        base += [{"name": "amountCents", "type": "Int!"}, {"name": "currency", "type": "String!"}, {"name": "state", "type": "String"}]
    else:
        base += [{"name": "name", "type": "String"}, {"name": "status", "type": "String"}, {"name": "description", "type": "String"}]
    if any(x in lower for x in ["post", "article", "ticket", "booking", "order", "claim", "report", "review"]):
        base += [{"name": "author", "type": "User"}, {"name": "authorId", "type": "ID"}]
    if any(x in lower for x in ["customer", "guest", "patient", "owner", "agent", "provider", "passenger", "user"]):
        base += [{"name": "name", "type": "String"}, {"name": "profile", "type": "String"}]
    return base[:8]


def _type_catalog(domain: str, types: list[str], rng: random.Random, cfg: WorldConfig) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for t in types:
        n_rel = rng.randint(cfg.min_relations, min(cfg.max_relations, max(len(types) - 1, 1)))
        pool = [x for x in types if x != t]
        neighbors = rng.sample(pool, k=min(n_rel, len(pool))) if pool else []
        aliases = ALIASES.get(t, [t.lower(), f"{t.lower()} record"])[:3]
        fields = _type_fields(t, domain)
        catalog[t] = {
            "type_name": t,
            "domain": domain,
            "aliases": aliases,
            "semantic_tags": [domain, "entity"],
            "neighbors": neighbors,
            "fields": fields,
            "description": f"{t} entity in the {domain} domain.",
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
        anchor_count = min(10, len(base_types), count)
        anchors = base_types[:anchor_count]
        remaining_pool = [t for t in base_types if t not in anchors]
        remaining_needed = max(count - len(anchors), 0)

        if count <= len(base_types):
            sampled = rng.sample(remaining_pool, k=min(remaining_needed, len(remaining_pool)))
            all_types = anchors + sampled
        else:
            # Keep names realistic; only append deterministic suffixes if configured count exceeds domain vocabulary.
            all_types = anchors + remaining_pool
            idx = 0
            while len(all_types) < count:
                all_types.append(f"{base_types[idx % len(base_types)]}Extension{idx + 1}")
                idx += 1

        catalog = _type_catalog(domain, all_types, rng, cfg)

        world_dir = worlds_dir / world_id
        ensure_dir(world_dir)
        write_json(world_dir / "type_catalog.json", {"world_id": world_id, "domain": domain, "types": catalog})

        sdl = _sdl_for_world(catalog)
        (world_dir / "schema.graphql").write_text(sdl, encoding="utf-8")

        field_catalog: list[dict] = []
        for t, meta in catalog.items():
            for field in meta.get("fields", []):
                coordinate = f"{t}.{field['name']}"
                doc_id = f"{world_id}:{coordinate}"
                field_record = {
                    "owner_type": t,
                    "field_name": field["name"],
                    "return_type": field.get("type", "String"),
                    "description": _field_description(t, field["name"], field.get("type", "String")),
                    "aliases": sorted(set(meta["aliases"] + [field["name"].replace("_", " "), field["name"].lower()])),
                    "path_to_root": _path_to_root(t, field["name"]),
                }
                field_catalog.append({"doc_id": doc_id, "coordinate": coordinate, **field_record})
                sdl_text = f"type {t} {{\n  {field['name']}: {field.get('type', 'String')}\n}}"
                corpus_rows.append(
                    CorpusRecord(
                        doc_id=doc_id,
                        coordinate=coordinate,
                        owner_type=t,
                        field_name=field["name"],
                        return_type=field.get("type", "String"),
                        short_text=f"Field {coordinate} in {domain}",
                        full_text=(
                            f"GraphQL field {coordinate} in domain {domain}. "
                            f"Owner aliases: {', '.join(meta['aliases']) or 'none'}. "
                            f"Related types: {', '.join(meta['neighbors']) or 'none'}. "
                            f"Field description: {_field_description(t, field['name'], field.get('type', 'String'))}"
                        ),
                        keywords_text=" | ".join(
                            sorted(
                                set(
                                    [coordinate.lower(), t.lower(), field["name"].lower(), domain]
                                    + [a.lower() for a in meta["aliases"]]
                                )
                            )
                        ),
                        description=_field_description(t, field["name"], field.get("type", "String")),
                        aliases=field_record["aliases"],
                        path_to_root=field_record["path_to_root"],
                        coordinate_text=coordinate,
                        field_signature_text=f"{coordinate}: {field.get('type', 'String')}",
                        field_semantic_text=(
                            f"GraphQL field {coordinate}. Owner type: {t}. Returns: {field.get('type', 'String')}. "
                            f"Description: {_field_description(t, field['name'], field.get('type', 'String'))}. "
                            f"Domain: {domain}. Related types: {', '.join(meta['neighbors']) or 'none'}."
                        ),
                        sdl_snippet_text=sdl_text,
                        retrieval_text=(
                            f"GraphQL field {coordinate}. Owner type: {t}. Returns: {field.get('type', 'String')}. "
                            f"Description: {_field_description(t, field['name'], field.get('type', 'String'))}. "
                            f"Domain: {domain}. Related types: {', '.join(meta['neighbors']) or 'none'}."
                        ),
                        metadata={
                            "world_id": world_id,
                            "domain": domain,
                            "neighbors": [f"{world_id}:{n}" for n in meta["neighbors"]],
                            "owner_aliases": meta["aliases"],
                        },
                    )
                )

        write_json(world_dir / "field_catalog.json", {"world_id": world_id, "domain": domain, "fields": field_catalog})

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
