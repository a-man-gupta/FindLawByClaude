from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "law_opinions"
EMBED_MODEL = "all-MiniLM-L6-v2"

_embedder: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    return _collection


def retrieve(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    global _collection
    embedder = get_embedder()
    query_embedding = embedder.encode(query).tolist()

    def _query(coll: chromadb.Collection) -> dict:
        return coll.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    try:
        results = _query(get_collection())
    except Exception:
        # Cached collection may have a stale HNSW reader if data was written
        # to disk after the handle was opened. Drop the cache and reopen.
        _collection = None
        results = _query(get_collection())

    output = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append({"text": doc, "metadata": meta, "distance": dist})

    return output


OPENAI_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "search_legal_database",
        "description": (
            "Search the local vector database of legal opinions from CourtListener. "
            "Use this to retrieve relevant case law, precedents, and legal text that "
            "can inform answers to legal questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing the legal topic or question.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of results to return. Defaults to 5.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}
