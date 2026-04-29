import json
import os
import chromadb
from model2vec import StaticModel

EMBEDDING_MODEL = os.path.join(os.path.dirname(__file__), "model")

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

    model = StaticModel.from_pretrained(EMBEDDING_MODEL)
    texts = [f"{m['title']}. {m['description']}" for m in movies]
    embeddings = model.encode(texts).tolist()

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


def semanticsearch(query: str, collection: chromadb.Collection, top_k: int = 3) -> list[dict]: # DEFAULT_TOP_RESULTS
    model = StaticModel.from_pretrained(EMBEDDING_MODEL)
    query_embedding = model.encode([query])[0].tolist()

    # ChromaDB returns a dict with parallel lists: ids, distances, metadatas…
    raw = collection.query(
        query_embeddings=[query_embedding],  # list of query vectors (we have one)
        n_results=top_k,
        include=["metadatas", "distances"],  # what fields to return alongside ids
    )

    """   
        L2/cosine Algs:
            raw["distances"][0] — ChromaDB returns cosine *distance* (0 = identical,
            2 = opposite).  We convert to a 0-100 similarity score:
            similarity = (1 - distance) * 100
            A distance of 0 → 100 %, distance of 1 → 0 %, distance > 1 → negative
            (clipped to 0 so we never show a negative score).
    """
    results = []
    for doc_id, meta, dist in zip(
        raw["ids"][0],            # list of matched document ids
        raw["metadatas"][0],      # list of metadata dicts
        raw["distances"][0],      # list of cosine distances
    ):
        similarity = round(max(0.0, (1 - dist) * 100), 1)
        results.append({
            "id":          doc_id,
            "title":       meta["title"],
            "year":        meta["year"],
            "genre":       meta["genre"],
            "description": meta["description"],
            "similarity":  similarity,
        })

    return results

def build_semantic_engine() -> tuple[chromadb.Collection, None]:
    with open("movies.json", "r", encoding="utf-8") as f: # LOAD MOVIES FROM .JSON
        movies = json.load(f)
    client = chromadb.Client()
    collection = build_index(movies, client)
    return collection


if __name__ == "__main__":
    col = build_semantic_engine()
    print(semanticsearch('story', col)) # EX. QUERY 