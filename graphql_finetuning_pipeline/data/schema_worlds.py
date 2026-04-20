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


# Cross-type shared-name pool. Every type in a synthetic world also gets a
# random subset of these so multiple owner types carry the same field name —
# which is what turns the pipeline's retrieval task into owner-type
# disambiguation rather than field-name guessing. Without this, synthetic
# worlds produce almost no ambiguity and the model has nothing to learn.
_SHARED_NAME_POOL: list[dict[str, str]] = [
    {"name": "actor", "type": "User"},
    {"name": "reporter", "type": "User"},
    {"name": "owner", "type": "User"},
    {"name": "assignee", "type": "User"},
    {"name": "approver", "type": "User"},
    {"name": "submittedBy", "type": "User"},
    {"name": "reviewer", "type": "User"},
    {"name": "recipient", "type": "User"},
    {"name": "initiator", "type": "User"},
    {"name": "lastModifiedBy", "type": "User"},
    {"name": "reference", "type": "String"},
    {"name": "externalId", "type": "String"},
    {"name": "channel", "type": "String"},
    {"name": "source", "type": "String"},
    {"name": "reason", "type": "String"},
    {"name": "note", "type": "String"},
    {"name": "tag", "type": "String"},
    {"name": "region", "type": "String"},
    {"name": "locale", "type": "String"},
]


# Hand-written type descriptions. The key trick for training owner-type
# disambiguation: each type's description has to describe *what the
# coordinates on that type represent*, not just repeat the type name.
# "A guest's booked stay at a property" is useful; "Reservation entity in
# hotels domain" is not. Missing entries fall back to a generic template
# but with the domain context baked in.
_TYPE_DESCRIPTIONS: dict[str, str] = {
    # hotels
    "Hotel": "A lodging property bookable by guests — has rooms, rates, and a location.",
    "Room": "An individual sleeping unit within a hotel; has a number, type, and nightly rate.",
    "Guest": "A person staying at a property; tied to bookings and loyalty accounts.",
    "Booking": "A tentative reservation hold, before payment and confirmation are finalised.",
    "Reservation": "A confirmed stay with dates, guests, and a rate lock — the post-payment record.",
    "Stay": "The actual check-in to check-out event, separate from the reservation intent.",
    "HotelReview": "A written review of a hotel or stay, posted by a guest after checkout.",
    "ServiceRequest": "A request for in-stay service (towels, maintenance, room service).",
    "IncidentReport": "A record of something going wrong during a stay (damage, complaint).",
    "RefundRequest": "A guest's request to reverse a charge on a reservation or stay.",
    "MaintenanceTicket": "An internal work order for property upkeep, not guest-facing.",
    "CheckInRequest": "A guest's digital check-in attempt, before keys are issued.",
    # shopping
    "Product": "An item offered for sale; the catalog row, independent of inventory or orders.",
    "Cart": "A shopper's in-progress selection, before checkout commits it to an order.",
    "Order": "A committed purchase — carts become orders after successful checkout.",
    "Customer": "The purchasing end-user account.",
    "Payment": "A financial transaction attempt tied to one order.",
    "Shipment": "A physical delivery batch that ships one or more order items.",
    "ProductReview": "A customer's written review of a product after purchase.",
    "ReturnRequest": "A customer's request to send back items from a completed order.",
    "Refund": "A completed reversal of funds to the customer's payment method.",
    "OrderStatusEvent": "A timestamped event in an order's lifecycle (placed, shipped, delivered).",
    "AbandonedCartEvent": "A cart that went idle without checkout; generated for re-engagement.",
    "FraudSignal": "A detected anomaly on an order indicating possible fraud.",
    "CouponRedemption": "A single use of a coupon code against an order.",
    # cars
    "Vehicle": "A specific car (VIN-identified), owned or listed by one party.",
    "VehicleModel": "A make/model/year definition — a template, not a specific car.",
    "VehicleListing": "A seller's listing of a specific vehicle for sale.",
    "OwnerProfile": "The registered owner of a specific vehicle; ties VIN to a person.",
    "ServiceRecord": "A dated service-shop maintenance entry for a vehicle.",
    "InspectionReport": "A documented inspection result with a pass/fail verdict.",
    "PriceQuote": "A dealer's quoted sale price for a listing at a point in time.",
    "TradeInOffer": "A dealer's proposed trade-in value for a customer's current vehicle.",
    "TestDriveBooking": "A customer's scheduled test drive appointment at a dealer.",
    "LeaseContract": "A signed lease agreement between a dealer and a lessee.",
    "AutoLoan": "A financing agreement against a specific vehicle purchase.",
    "RecallNotice": "A manufacturer recall announcement affecting a vehicle model or range.",
    "RepairEstimate": "A shop's written estimate for repair work on a vehicle.",
}


def _describe_type(type_name: str, domain: str) -> str:
    if type_name in _TYPE_DESCRIPTIONS:
        return _TYPE_DESCRIPTIONS[type_name]
    # Fallback keeps domain context so even uncatalogued types disambiguate.
    return f"{type_name}: an entity in the {domain} domain (auto-described)."


def _type_fields(type_name: str, domain: str, rng: random.Random) -> list[dict[str, str]]:
    """Emit fields for one type, biased toward cross-type ambiguity.

    The returned list always includes: id, createdAt, updatedAt, a small set
    of role-specific fields keyed on the type name, PLUS 2-3 fields sampled
    from the shared-name pool so multiple owner types carry the same field
    name within one world. That shared-name overlap is the prerequisite for
    the competition-set prompt in openai_seed._user_prompt_competition.
    """
    lower = type_name.lower()
    base: list[dict[str, str]] = [
        {"name": "id", "type": "ID!"},
        {"name": "createdAt", "type": "String"},
        {"name": "updatedAt", "type": "String"},
    ]
    if any(x in lower for x in ["hotel", "property", "facility"]):
        base += [{"name": "displayName", "type": "String!"}, {"name": "city", "type": "String"}, {"name": "rating", "type": "Float"}]
    elif any(x in lower for x in ["room", "seat", "ticket"]):
        base += [{"name": "label", "type": "String!"}, {"name": "status", "type": "String"}, {"name": "priceCents", "type": "Int"}]
    elif any(x in lower for x in ["order", "reservation", "booking", "invoice", "claim"]):
        base += [{"name": "status", "type": "String!"}, {"name": "totalCents", "type": "Int"}, {"name": "currency", "type": "String"}]
    elif any(x in lower for x in ["customer", "guest", "patient", "owner", "agent", "provider", "passenger"]):
        base += [{"name": "displayName", "type": "String"}, {"name": "email", "type": "String"}, {"name": "phone", "type": "String"}]
    elif any(x in lower for x in ["payment", "transaction", "transfer", "payout", "refund"]):
        base += [{"name": "amountCents", "type": "Int!"}, {"name": "currency", "type": "String!"}, {"name": "state", "type": "String"}]
    else:
        base += [{"name": "displayName", "type": "String"}, {"name": "status", "type": "String"}, {"name": "description", "type": "String"}]
    if any(x in lower for x in ["post", "article", "ticket", "booking", "order", "claim", "report", "review"]):
        base += [{"name": "author", "type": "User"}, {"name": "authorId", "type": "ID"}]

    # Shared-name pool: inject 2-3 names at random so cross-type ambiguity is
    # the norm rather than the exception. Seeded per-type via rng so the
    # choice is stable for a given world.
    already = {f["name"] for f in base}
    pool = [p for p in _SHARED_NAME_POOL if p["name"] not in already]
    rng.shuffle(pool)
    base += pool[: rng.randint(2, 3)]

    return base[:10]


def _type_catalog(domain: str, types: list[str], rng: random.Random, cfg: WorldConfig) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for t in types:
        n_rel = rng.randint(cfg.min_relations, min(cfg.max_relations, max(len(types) - 1, 1)))
        pool = [x for x in types if x != t]
        neighbors = rng.sample(pool, k=min(n_rel, len(pool))) if pool else []
        aliases = ALIASES.get(t, [t.lower(), f"{t.lower()} record"])[:3]
        fields = _type_fields(t, domain, rng)
        catalog[t] = {
            "type_name": t,
            "domain": domain,
            "aliases": aliases,
            "semantic_tags": [domain, "entity"],
            "neighbors": neighbors,
            "fields": fields,
            "description": _describe_type(t, domain),
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
                    "owner_description": meta.get("description", ""),
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
