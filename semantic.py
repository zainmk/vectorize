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

# Path to the raw movies data (same file used by main.py / bm25.py).
MOVIES_FILE = "movies.json"

# Directory where ChromaDB will persist its SQLite + vector files.
# Delete this folder to force a full re-index on next run.
CHROMA_PERSIST_DIR = "./chroma_store"

# Name of the collection inside ChromaDB (like a table name in a database).
COLLECTION_NAME = "movies"

# The sentence-transformer model to use for embedding text.
# Must be the same model at index time AND query time so vectors are
# comparable.  Changing this requires deleting CHROMA_PERSIST_DIR and
# re-indexing.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many results to return from a semantic search by default.
DEFAULT_TOP_K = 5


# ---------------------------------------------------------------------------
# Step 1 — Load the raw data
# ---------------------------------------------------------------------------

def load_movies(path: str = MOVIES_FILE) -> list[dict]:
    """Read movies.json and return the list of movie dicts."""
    with open(path, "r", encoding="utf-8") as f:
        movies = json.load(f)  # parse the JSON array into a Python list
    return movies


# ---------------------------------------------------------------------------
# Step 2 — Build (or reuse) the ChromaDB collection
# ---------------------------------------------------------------------------

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Return a ChromaDB client that persists data to CHROMA_PERSIST_DIR.
    On first run this creates the directory; on subsequent runs it reuses
    whatever is already stored there.
    """
    # PersistentClient saves to disk automatically after every operation.
    # Use chromadb.Client() instead if you want an in-memory-only store.
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client


def build_index(movies: list[dict], client: chromadb.PersistentClient) -> chromadb.Collection:
    """
    Create (or load) the ChromaDB collection and populate it with movie
    embeddings.  If the collection already contains documents we skip
    re-embedding to avoid unnecessary work. 

    ChromaDB stores three things per document:
      - id        : a unique string key
      - document  : the raw text that was embedded
      - metadata  : arbitrary key/value pairs (title, year, genre, …)

    Parameters
    ----------
    movies : list of movie dicts loaded from movies.json
    client : an active ChromaDB client

    Returns
    -------
    A ChromaDB Collection object ready for querying.
    """

    # get_or_create_collection returns an existing collection if one with
    # this name already exists, otherwise it creates a fresh one.
    # cosine similarity is the right distance metric for sentence embeddings
    # because the vectors are L2-normalised — direction matters, not magnitude.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # use cosine distance internally
    )

    # If the collection already has documents we assume the index is current
    # and skip re-embedding.  For production code you'd want a smarter
    # freshness check (e.g. comparing a hash of movies.json).
    if collection.count() == len(movies):
        print(f"[semantic] Collection '{COLLECTION_NAME}' already indexed "
              f"({collection.count()} documents) — skipping re-embed.")
        return collection

    print(f"[semantic] Embedding {len(movies)} movies with '{EMBEDDING_MODEL}'…")

    # Load the sentence-transformer model.
    # The first call downloads the model weights (~90 MB) to a local cache;
    # subsequent calls load from that cache instantly.
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Build the text we want to embed for each movie.
    # Concatenating title + description gives the model richer context than
    # embedding the description alone.
    texts = [
        f"{m['title']}. {m['description']}"
        for m in movies
    ]

    # Embed all texts in one batched call — much faster than a loop.
    # Returns a numpy array of shape (len(movies), 384).
    embeddings = model.encode(texts, show_progress_bar=True)

    # ChromaDB expects Python lists, not numpy arrays, so we convert.
    embeddings_list = embeddings.tolist()

    # Each document needs a unique string ID.
    ids = [str(m["id"]) for m in movies]

    # Metadata is stored alongside the vector and returned with results.
    # We store everything we need to render a result card so we never have
    # to do a secondary lookup into movies.json at query time.
    metadatas = [
        {
            "title":       m["title"],
            "year":        m["year"],
            "genre":       m["genre"],
            "description": m["description"],
        }
        for m in movies
    ]

    # upsert — insert if new, replace if the id already exists.
    # This means if you add movies to movies.json and re-run, only the new
    # ones will be inserted (as long as their ids are unique).
    collection.upsert(
        ids=ids,
        embeddings=embeddings_list,
        documents=texts,       # the raw text (optional but useful for debugging)
        metadatas=metadatas,
    )

    print(f"[semantic] Indexed {len(movies)} movies successfully.")
    return collection


# ---------------------------------------------------------------------------
# Step 3 — Query
# ---------------------------------------------------------------------------

def search(query: str, collection: chromadb.Collection, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """
    Embed the user's query and find the most similar movies in the collection.

    Parameters
    ----------
    query      : free-text search string from the user
    collection : the indexed ChromaDB collection
    top_k      : maximum number of results to return

    Returns
    -------
    A list of result dicts, each containing:
        id, title, year, genre, description, similarity (0-100 float)
    """

    # Load the same model used at index time so our query vector lives in
    # the same embedding space as the stored document vectors.
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed the single query string.  encode() always returns an array so we
    # take [0] to get the 1-D vector, then convert to a plain Python list.
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
    """
    Convenience function called at server startup (e.g. from main.py's
    lifespan handler).  Returns the ready-to-query collection.

    Usage in main.py:
        from semantic import build_semantic_engine, search as semantic_search
        collection = build_semantic_engine()
        results = semantic_search("lonely AI", collection)
    """
    movies     = load_movies()          # load movies.json
    client     = get_chroma_client()    # open / create the on-disk store
    collection = build_index(movies, client)  # embed & index (or reuse cache)
    return collection


# ---------------------------------------------------------------------------
# CLI — run this file directly to test without starting FastAPI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build (or reload) the index.
    col = build_semantic_engine()

    print("\nSemantic movie search — type a query, hit Enter (Ctrl+C to quit)\n")

    while True:
        try:
            q = input("Query: ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if not q:
            continue  # ignore empty input

        hits = search(q, col)

        if not hits:
            print("  No results.\n")
            continue

        for rank, h in enumerate(hits, start=1):
            # Print a compact result line for quick visual inspection.
            print(f"  {rank}. [{h['similarity']:5.1f}%]  {h['title']} ({h['year']}) — {h['genre']}")
            print(f"       {h['description'][:100]}…\n")
