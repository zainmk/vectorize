"""
semantic.py — ChromaDB-backed semantic search over the movies dataset.

How it works at a high level:
  1. Each movie's title + description is turned into a dense vector (embedding)
     by a small sentence-transformer model running locally.
  2. Those vectors are stored in a ChromaDB collection persisted to disk so
     we only pay the embedding cost once.
  3. At query time the user's query string is embedded the same way, and
     ChromaDB finds the nearest movie vectors using cosine similarity.

Dependencies:
    pip install chromadb sentence-transformers
"""

import json
import chromadb

# SentenceTransformer turns text into a fixed-length numeric vector.
# 'all-MiniLM-L6-v2' is a lightweight but capable model (384 dimensions).
# It runs entirely locally — no API key or internet required at query time.
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The sentence-transformer model to use for embedding text.
# Must be the same model at index time AND query time so vectors are
# comparable.  Changing this requires deleting CHROMA_PERSIST_DIR and
# re-indexing.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Step 2 — Build (or reuse) the ChromaDB collection
# ---------------------------------------------------------------------------


def build_index(movies: list[dict], client: chromadb.Client) -> chromadb.Collection:
    """
    ChromaDB stores three things per document:
      - id        : a unique string key
      - document  : the raw text that was embedded
      - metadata  : arbitrary key/value pairs (title, year, genre, …)
    """

    # using cosine similiarity to compare the angles between vectors, as sentence-embedding vectors are all L2 normalized (all equal to 'one' ~ same length)
    collection = client.create_collection(name='movies', metadata={"hnsw:space": "cosine"})

    print(f"[semantic] Embedding {len(movies)} movies with '{EMBEDDING_MODEL}'…")

    model = SentenceTransformer(EMBEDDING_MODEL) # loading the model gets cached
    texts = [f"{m['title']}. {m['description']}" for m in movies] # encode full data (title + desc.)

    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    ids = [str(m["id"]) for m in movies]
    metadatas = [
        {
            "title":       m["title"],
            "year":        m["year"],
            "genre":       m["genre"],
            "description": m["description"],
        }
        for m in movies
    ]
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection


# ---------------------------------------------------------------------------
# Step 3 — Query
# ---------------------------------------------------------------------------

def semanticsearch(query: str, collection: chromadb.Collection, top_k: int = 10) -> list[dict]:
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].tolist()
    # query() finds the top_k nearest neighbours to our query vector.
    # ChromaDB returns a dict with parallel lists: ids, distances, metadatas…
    raw = collection.query(
        query_embeddings=[query_embedding],  # list of query vectors (we have one)
        n_results=top_k,
        include=["metadatas", "distances"],  # what fields to return alongside ids
    )

    # raw["distances"][0] — ChromaDB returns cosine *distance* (0 = identical,
    # 2 = opposite).  We convert to a 0-100 similarity score:
    #   similarity = (1 - distance) * 100
    # A distance of 0 → 100 %, distance of 1 → 0 %, distance > 1 → negative
    # (clipped to 0 so we never show a negative score).
    results = []
    for doc_id, meta, dist in zip(
        raw["ids"][0],            # list of matched document ids
        raw["metadatas"][0],      # list of metadata dicts
        raw["distances"][0],      # list of cosine distances
    ):
        similarity = round(max(0.0, (1 - dist) * 100), 1)  # convert & clip
        results.append({
            "id":          doc_id,
            "title":       meta["title"],
            "year":        meta["year"],
            "genre":       meta["genre"],
            "description": meta["description"],
            "similarity":  similarity,
        })

    return results


# ---------------------------------------------------------------------------
# Step 4 — Wire everything together
# ---------------------------------------------------------------------------

def build_semantic_engine() -> tuple[chromadb.Collection, None]:
    # LOAD MOVIES FROM JSON
    with open("movies.json", "r", encoding="utf-8") as f:
        movies = json.load(f)
    client = chromadb.Client()
    collection = build_index(movies, client)
    return collection


# ---------------------------------------------------------------------------
# CLI — run this file directly to test without starting FastAPI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    col = build_semantic_engine()
    print(semanticsearch('story', col)) # EX. QUERY 