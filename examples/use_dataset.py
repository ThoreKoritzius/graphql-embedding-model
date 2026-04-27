"""
Example: load and explore xthor/Qwen3-Embedding-GraphQL-v1

Shows how to load each split (train / val / test / corpus) and run a
quick retrieval eval loop using the published model.
"""

import json
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import numpy as np

REPO = "xthor/Qwen3-Embedding-GraphQL-v1"
MODEL = "xthor/Qwen3-Embedding-0.6B-GraphQL"
REPO_TYPE = "dataset"


def load_jsonl(filename: str) -> list[dict]:
    path = hf_hub_download(REPO, filename, repo_type=REPO_TYPE)
    with open(path) as f:
        return [json.loads(line) for line in f]


# --- 1. inspect a few training rows ---
print("=== train sample ===")
train = load_jsonl("train.jsonl")
for row in train[:3]:
    print(f"  query:    {row['query']!r}")
    print(f"  positive: {row['positive_coordinate']}")
    print(f"  negs:     {row['negative_coordinates'][:3]}")
    print()

# --- 2. load the corpus (metadata field excluded for compatibility) ---
print("=== corpus size ===")
corpus_rows = load_jsonl("corpus.jsonl")
print(f"  {len(corpus_rows)} fields")
print(f"  sample: {corpus_rows[0]['coordinate']}  —  {corpus_rows[0]['retrieval_text'][:80]}")
print()

# --- 3. small retrieval eval on the test split ---
print("=== retrieval eval (first 20 test rows) ===")
test = load_jsonl("test.jsonl")[:20]

model = SentenceTransformer(MODEL)

corpus_texts = [r["retrieval_text"] for r in corpus_rows]
coord_index = {r["coordinate"]: i for i, r in enumerate(corpus_rows)}

corpus_embs = model.encode(corpus_texts, prompt_name="document", normalize_embeddings=True,
                            batch_size=256, show_progress_bar=True)

hits_at_1 = 0
for row in test:
    q_emb = model.encode(row["query"], prompt_name="query", normalize_embeddings=True)
    scores = (q_emb @ corpus_embs.T)
    top1_idx = int(np.argmax(scores))
    predicted = corpus_rows[top1_idx]["coordinate"]
    correct = predicted == row["positive_coordinate"]
    hits_at_1 += correct
    status = "OK" if correct else "MISS"
    print(f"  [{status}] {row['query']!r}")
    print(f"         expected={row['positive_coordinate']}  got={predicted}")

print(f"\nexact_match@1 on first 20 rows: {hits_at_1}/{len(test)} = {hits_at_1/len(test):.2f}")
