import asyncio
import hashlib
import re
from collections.abc import AsyncIterator
from typing import Any

import chromadb
import httpx
from sentence_transformers import SentenceTransformer

COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4/opinions/"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "law_opinions"
EMBED_MODEL = "all-MiniLM-L6-v2"

SEED_QUERIES = [
    "contract breach",
    "first amendment freedom of speech",
    "miranda rights criminal procedure",
    "copyright infringement fair use",
    "employment discrimination civil rights",
    "fourth amendment search seizure",
    "negligence tort liability",
    "due process constitutional rights",
    "intellectual property trademark",
    "habeas corpus criminal defense",
]

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


async def fetch_courtlistener(query: str, num_results: int = 50) -> list[dict[str, Any]]:
    params = {"format": "json", "search": query, "page_size": num_results}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(COURTLISTENER_BASE, params=params)
        response.raise_for_status()
        data = response.json()

    docs = []
    for item in data.get("results", []):
        raw_text = item.get("plain_text") or item.get("html") or ""
        text = re.sub(r"<[^>]+>", " ", raw_text).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        cluster = item.get("cluster") or {}
        docs.append(
            {
                "id": str(item.get("id", "")),
                "case_name": item.get("case_name") or cluster.get("case_name", "Unknown"),
                "court": cluster.get("court", "Unknown") if isinstance(cluster, dict) else "Unknown",
                "date_filed": item.get("date_filed") or item.get("date_created", ""),
                "text": text,
                "source_url": f"https://www.courtlistener.com{item.get('absolute_url', '')}",
            }
        )
    return docs


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def ingest_to_chroma(docs: list[dict[str, Any]], collection_name: str = COLLECTION_NAME) -> int:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )
    embedder = get_embedder()

    all_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    all_documents: list[str] = []
    all_metadatas: list[dict[str, Any]] = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{doc['id']}_{i}".encode()).hexdigest()
            all_ids.append(chunk_id)
            all_embeddings.append(embedder.encode(chunk).tolist())
            all_documents.append(chunk)
            all_metadatas.append(
                {
                    "case_name": doc["case_name"],
                    "court": doc["court"],
                    "date_filed": doc["date_filed"],
                    "source_url": doc["source_url"],
                    "opinion_id": doc["id"],
                    "chunk_index": i,
                }
            )

    if not all_ids:
        return 0

    for start in range(0, len(all_ids), 100):
        end = start + 100
        collection.upsert(
            ids=all_ids[start:end],
            embeddings=all_embeddings[start:end],
            documents=all_documents[start:end],
            metadatas=all_metadatas[start:end],
        )

    return len(all_ids)


def get_db_stats() -> dict[str, int]:
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        return {"total_chunks": collection.count()}
    except Exception:
        return {"total_chunks": 0}


async def run_ingestion(queries: list[str] | None = None) -> AsyncIterator[dict[str, Any]]:
    """Async generator yielding progress events for SSE streaming."""
    queries = queries or SEED_QUERIES
    total_chunks = 0

    yield {"event": "start", "total_queries": len(queries)}

    for i, query in enumerate(queries):
        yield {"event": "query_start", "query": query, "index": i, "total": len(queries)}
        try:
            docs = await fetch_courtlistener(query, num_results=50)
            yield {"event": "fetched", "query": query, "docs": len(docs)}

            if docs:
                chunks = ingest_to_chroma(docs)
                total_chunks += chunks
                yield {"event": "ingested", "query": query, "chunks": chunks, "total_chunks": total_chunks}
            else:
                yield {"event": "ingested", "query": query, "chunks": 0, "total_chunks": total_chunks}

        except httpx.HTTPStatusError as e:
            yield {"event": "error", "query": query, "detail": f"HTTP {e.response.status_code}"}
        except Exception as e:
            yield {"event": "error", "query": query, "detail": str(e)}

    yield {"event": "complete", "total_chunks": total_chunks, "total_queries": len(queries)}


async def main() -> None:
    try:
        from tqdm import tqdm  # type: ignore[import]
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    queries = SEED_QUERIES
    total = len(queries)
    bar = tqdm(total=total, unit="query") if has_tqdm else None
    total_chunks = 0

    async for event in run_ingestion(queries):
        ev = event["event"]
        if ev == "start":
            print(f"Starting ingestion of {event['total_queries']} queries...\n")
        elif ev == "query_start":
            if bar:
                bar.set_description(f"Fetching: {event['query']!r}")
            else:
                print(f"[{event['index']+1}/{event['total']}] Fetching: {event['query']!r}")
        elif ev == "fetched":
            if not bar:
                print(f"  Retrieved {event['docs']} opinions")
        elif ev == "ingested":
            total_chunks = event["total_chunks"]
            if bar:
                bar.update(1)
                bar.set_postfix(chunks=total_chunks)
            else:
                print(f"  Ingested {event['chunks']} chunks (total: {total_chunks})")
        elif ev == "error":
            msg = f"  ERROR on {event['query']!r}: {event['detail']}"
            if bar:
                bar.write(msg)
            else:
                print(msg)
        elif ev == "complete":
            if bar:
                bar.close()
            print(f"\nDone! Total chunks stored: {event['total_chunks']}")


if __name__ == "__main__":
    asyncio.run(main())
