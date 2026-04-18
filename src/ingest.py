import asyncio
import hashlib
import os
import re
from collections.abc import AsyncIterator
from typing import Any

import chromadb
import httpx
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "law_opinions"
EMBED_MODEL = "all-MiniLM-L6-v2"

COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4/opinions/"

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


# ── Source: HuggingFace public legal datasets (no auth) ──────────────────────

def fetch_hf_legal_cases(max_docs: int = 500) -> list[dict[str, Any]]:
    """
    Streams from joelniklaus/legal_case_reports (AU cases, free, no auth).
    Falls back to eureka-moment dataset if unavailable.
    """
    from datasets import load_dataset  # type: ignore[import]

    docs = []
    try:
        ds = load_dataset(
            "joelniklaus/legal_case_reports",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        for item in ds:
            if len(docs) >= max_docs:
                break
            text = (item.get("text") or item.get("catchphrases") or "").strip()
            if not text or len(text) < 100:
                continue
            docs.append(
                {
                    "id": f"hf_lcr_{hashlib.md5(text[:80].encode()).hexdigest()[:10]}",
                    "case_name": item.get("name", "Unknown Case"),
                    "court": item.get("court", "Australian Court"),
                    "date_filed": str(item.get("year", "")),
                    "text": text,
                    "source_url": item.get("url", ""),
                }
            )
    except Exception:
        pass

    if not docs:
        # Fallback: lex_glue SCOTUS dataset
        try:
            ds = load_dataset("lex_glue", "scotus", split="train", streaming=True, trust_remote_code=False)
            labels = [
                "Criminal Procedure", "Civil Rights", "First Amendment", "Due Process",
                "Privacy", "Attorneys", "Unions", "Economic Activity", "Judicial Power",
                "Federalism", "Interstate Relations", "Federal Taxation", "Miscellaneous",
                "Private Action",
            ]
            for item in ds:
                if len(docs) >= max_docs:
                    break
                text = (item.get("text") or "").strip()
                if not text or len(text) < 100:
                    continue
                label_idx = item.get("label", 12)
                area = labels[label_idx] if label_idx < len(labels) else "Unknown"
                docs.append(
                    {
                        "id": f"hf_scotus_{hashlib.md5(text[:80].encode()).hexdigest()[:10]}",
                        "case_name": f"SCOTUS Opinion ({area})",
                        "court": "U.S. Supreme Court",
                        "date_filed": "",
                        "text": text,
                        "source_url": "",
                    }
                )
        except Exception:
            pass

    return docs


def fetch_hf_us_opinions(topic: str, max_docs: int = 100) -> list[dict[str, Any]]:
    """
    Streams MultiLegalPile (English US legal text subset), keyword-filtered.
    """
    from datasets import load_dataset  # type: ignore[import]

    docs = []
    keywords = topic.lower().split()
    try:
        ds = load_dataset(
            "pile-of-law/pile-of-law",
            "r_legaladvice",
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        for item in ds:
            if len(docs) >= max_docs:
                break
            text = (item.get("text") or "").strip()
            if not text or len(text) < 200:
                continue
            if not any(kw in text.lower() for kw in keywords):
                continue
            docs.append(
                {
                    "id": f"pol_{hashlib.md5(text[:80].encode()).hexdigest()[:10]}",
                    "case_name": f"Legal Text: {topic.title()}",
                    "court": "Various",
                    "date_filed": "",
                    "text": text[:5000],
                    "source_url": "",
                }
            )
    except Exception:
        pass
    return docs


# ── Source: Oyez API (SCOTUS, completely free, no auth) ──────────────────────

# Strategic year sampling — captures landmark cases across major doctrinal eras.
# Civil rights era, modern speech doctrine, fair use, privacy, recent SCOTUS.
OYEZ_TERMS = [
    "1953", "1962", "1963", "1965", "1966", "1967", "1968", "1969", "1971",
    "1972", "1973", "1976", "1978", "1984", "1988", "1989", "1992", "1994",
    "1997", "2000", "2003", "2010", "2014", "2015", "2017", "2018", "2019",
    "2020", "2021", "2022", "2023",
]


async def fetch_oyez_cases(max_docs: int = 300, per_term_cap: int = 12) -> list[dict[str, Any]]:
    """Fetch SCOTUS cases from Oyez API — free, no auth, rich opinion text.

    Phase 1: gather case lists across landmark terms in parallel.
    Phase 2: fetch detail pages in parallel (with concurrency limit).
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        list_sem = asyncio.Semaphore(8)
        detail_sem = asyncio.Semaphore(20)

        async def fetch_term_list(term: str) -> list[dict]:
            async with list_sem:
                try:
                    resp = await client.get(
                        "https://api.oyez.org/cases",
                        params={"filter": f"term:{term}", "per_page": 100},
                    )
                    resp.raise_for_status()
                    cases = resp.json()
                    return cases[:per_term_cap]  # spread coverage across terms
                except Exception:
                    return []

        term_results = await asyncio.gather(*(fetch_term_list(t) for t in OYEZ_TERMS))
        all_case_metas: list[dict] = []
        for cases in term_results:
            all_case_metas.extend(cases)
        all_case_metas = all_case_metas[:max_docs]

        async def fetch_detail(case: dict) -> tuple[dict, dict | None]:
            href = case.get("href", "")
            if not href:
                return case, None
            async with detail_sem:
                try:
                    detail_resp = await client.get(href)
                    detail_resp.raise_for_status()
                    return case, detail_resp.json()
                except Exception:
                    return case, None

        results = await asyncio.gather(*(fetch_detail(c) for c in all_case_metas))

    docs = []
    for case_meta, detail in results:
        if detail is None:
            continue

        parts = []
        if detail.get("facts_of_the_case"):
            parts.append(f"FACTS: {re.sub(r'<[^>]+>', ' ', detail['facts_of_the_case'])}")
        if detail.get("question"):
            parts.append(f"QUESTION: {re.sub(r'<[^>]+>', ' ', detail['question'])}")
        if detail.get("conclusion"):
            parts.append(f"CONCLUSION: {re.sub(r'<[^>]+>', ' ', detail['conclusion'])}")
        if detail.get("description"):
            parts.append(f"DESCRIPTION: {detail['description']}")

        text = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if len(text) < 100:
            continue

        citation = detail.get("citation") or {}
        year = citation.get("year") or str(detail.get("term", ""))
        href = case_meta.get("href", "")
        docs.append(
            {
                "id": f"oyez_{detail.get('ID', hashlib.md5(href.encode()).hexdigest()[:8])}",
                "case_name": detail.get("name", case_meta.get("name", "Unknown")),
                "court": "U.S. Supreme Court",
                "date_filed": year,
                "text": text,
                "source_url": detail.get("justia_url") or href,
            }
        )

    return docs


# ── Source: CourtListener (requires free token) ───────────────────────────────

async def fetch_courtlistener(query: str, num_results: int = 50) -> list[dict[str, Any]]:
    token = os.environ.get("COURTLISTENER_TOKEN")
    if not token:
        raise ValueError("COURTLISTENER_TOKEN not set")

    headers = {"Authorization": f"Token {token}"}
    params = {"format": "json", "search": query, "page_size": num_results}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(COURTLISTENER_BASE, params=params, headers=headers)
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
                "id": f"cl_{item.get('id', '')}",
                "case_name": item.get("case_name") or cluster.get("case_name", "Unknown"),
                "court": cluster.get("court", "Unknown") if isinstance(cluster, dict) else "Unknown",
                "date_filed": item.get("date_filed") or item.get("date_created", ""),
                "text": text,
                "source_url": f"https://www.courtlistener.com{item.get('absolute_url', '')}",
            }
        )
    return docs


# ── Chunking + ChromaDB ───────────────────────────────────────────────────────

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
        for i, chunk in enumerate(chunk_text(doc["text"])):
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


# ── Ingestion orchestrator ────────────────────────────────────────────────────

async def run_ingestion(queries: list[str] | None = None) -> AsyncIterator[dict[str, Any]]:
    queries = queries or SEED_QUERIES
    total_chunks = 0
    has_cl = bool(os.environ.get("COURTLISTENER_TOKEN"))

    total_steps = 1 + (len(queries) if has_cl else 0)

    yield {
        "event": "start",
        "total_queries": total_steps,
        "sources": {"oyez": True, "courtlistener": has_cl},
    }

    # Step 1: Oyez SCOTUS landmark cases (PRIMARY — always run)
    yield {"event": "query_start", "query": "Oyez SCOTUS landmark cases", "index": 0, "total": total_steps}
    try:
        oyez_docs = await fetch_oyez_cases(max_docs=300)
        yield {"event": "fetched", "query": "Oyez SCOTUS landmark cases", "docs": len(oyez_docs), "source": "Oyez"}
        if oyez_docs:
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, ingest_to_chroma, oyez_docs)
            total_chunks += chunks
            yield {"event": "ingested", "query": "Oyez SCOTUS landmark cases", "chunks": chunks, "total_chunks": total_chunks}
        else:
            yield {"event": "skipped", "query": "Oyez SCOTUS landmark cases", "detail": "Oyez returned no documents"}
    except Exception as e:
        yield {"event": "error", "query": "Oyez SCOTUS landmark cases", "detail": str(e)}

    # Step 2: per-query CourtListener supplementation (only if token available)
    if has_cl:
        for i, query in enumerate(queries):
            yield {"event": "query_start", "query": query, "index": i + 1, "total": total_steps}
            docs: list[dict] = []
            try:
                docs = await fetch_courtlistener(query, num_results=50)
                yield {"event": "fetched", "query": query, "docs": len(docs), "source": "CourtListener"}
            except Exception as e:
                yield {"event": "warning", "query": query, "detail": str(e)}

            if docs:
                loop = asyncio.get_event_loop()
                chunks = await loop.run_in_executor(None, ingest_to_chroma, docs)
                total_chunks += chunks
                yield {"event": "ingested", "query": query, "chunks": chunks, "total_chunks": total_chunks}
            else:
                yield {"event": "skipped", "query": query, "detail": "No documents fetched"}
    else:
        yield {
            "event": "info",
            "query": "CourtListener",
            "detail": "No COURTLISTENER_TOKEN set — Oyez-only ingestion. Add token to .env to expand corpus.",
        }

    yield {"event": "complete", "total_chunks": total_chunks, "total_queries": total_steps}


async def main() -> None:
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    total_steps = len(SEED_QUERIES) + 1
    bar = tqdm(total=total_steps, unit="step") if has_tqdm else None
    total_chunks = 0

    async for event in run_ingestion():
        ev = event["event"]
        if ev == "start":
            sources = event.get("sources", {})
            cl = "✓" if sources.get("courtlistener") else "✗ (add COURTLISTENER_TOKEN for more)"
            print(f"CourtListener: {cl} | HuggingFace: ✓\n")
        elif ev == "query_start":
            if bar:
                bar.set_description(event["query"][:40])
            else:
                print(f"[{event['index']+1}/{event['total']}] {event['query']}")
        elif ev == "fetched":
            msg = f"  {event['source']}: {event['docs']} docs"
            if bar:
                bar.write(msg)
            else:
                print(msg)
        elif ev == "ingested":
            total_chunks = event["total_chunks"]
            if bar:
                bar.update(1)
                bar.set_postfix(chunks=total_chunks)
            else:
                print(f"  → {event['chunks']} chunks (total: {total_chunks})")
        elif ev == "skipped":
            if bar:
                bar.update(1)
                bar.write(f"  ⟳ {event['query']}: {event['detail']}")
            else:
                print(f"  SKIP: {event['detail']}")
        elif ev in ("error", "warning"):
            msg = f"  {'ERR' if ev == 'error' else 'WARN'}: {event.get('detail', '')}"
            if bar:
                bar.write(msg)
                if ev == "error":
                    bar.update(1)
            else:
                print(msg)
        elif ev == "complete":
            if bar:
                bar.close()
            print(f"\nDone. Total chunks: {event['total_chunks']}")
            print("\nTo add more data:")
            print("  1. Register free at courtlistener.com → add COURTLISTENER_TOKEN to .env")
            print("  2. Register free at case.law → add CAP_TOKEN to .env")
            print("  3. Run python src/ingest.py again")


if __name__ == "__main__":
    asyncio.run(main())
