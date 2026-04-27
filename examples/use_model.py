"""
Example: semantic field retrieval with xthor/Qwen3-Embedding-0.6B-GraphQL

Encodes a natural-language query and a small corpus of GraphQL field
descriptions, then ranks them by cosine similarity.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("xthor/Qwen3-Embedding-0.6B-GraphQL")

# --- corpus: one entry per Type.field coordinate ---
corpus = [
    "GraphQL field Room.priceCents. Owner type: Room. Returns: Int.",
    "GraphQL field RoomUpgradeOffer.priceCents. Owner type: RoomUpgradeOffer. Returns: Int.",
    "GraphQL field Ticket.priceCents. Owner type: Ticket. Returns: Int.",
    "GraphQL field Booking.checkInDate. Owner type: Booking. Returns: String.",
    "GraphQL field SlaPolicy.description. Owner type: SlaPolicy. Returns: String.",
    "GraphQL field Incident.description. Owner type: Incident. Returns: String.",
]

queries = [
    "What's the nightly rate for this room?",
    "I need to understand what commitments we have regarding support response times.",
]

# encode corpus once; re-use for all queries
corpus_embeddings = model.encode(corpus, prompt_name="document", normalize_embeddings=True)

for query in queries:
    q_emb = model.encode(query, prompt_name="query", normalize_embeddings=True)
    scores = (q_emb @ corpus_embeddings.T).tolist()

    ranked = sorted(zip(scores, corpus), reverse=True)

    print(f"Query: {query!r}")
    for rank, (score, field) in enumerate(ranked, 1):
        print(f"  {rank}. {score:.3f}  {field}")
    print()
