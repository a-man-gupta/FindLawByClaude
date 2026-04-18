import asyncio
import json
import os

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .ingest import SEED_QUERIES, get_db_stats, run_ingestion
from .rag import SEARCH_LEGAL_DATABASE_TOOL, retrieve

load_dotenv()

app = FastAPI(title="FindLawByClaude")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
SYSTEM_PROMPT = (
    "You are FindLawByClaude, an AI legal research assistant. "
    "You have access to a database of real US court opinions. "
    "When answering legal questions, always search the database first to ground your answer in actual case law. "
    "Cite specific cases by name when they are relevant. "
    "Format your response with clear sections using markdown: bold key terms, use bullet points for lists of factors or elements. "
    "Be clear that you are not providing legal advice and users should consult a licensed attorney for their specific situation."
)

_ingest_lock = asyncio.Lock()
_ingest_running = False


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    case_name: str
    court: str
    date_filed: str
    source_url: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestRequest(BaseModel):
    queries: list[str] | None = None


def _build_tool_result_content(results: list[dict]) -> str:
    if not results:
        return "No relevant cases found in the database."
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        parts.append(
            f"[{i}] Case: {meta.get('case_name', 'Unknown')}\n"
            f"    Court: {meta.get('court', 'Unknown')}\n"
            f"    Date: {meta.get('date_filed', 'Unknown')}\n"
            f"    URL: {meta.get('source_url', '')}\n"
            f"    Excerpt: {r['text'][:800]}"
        )
    return "\n\n".join(parts)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = [{"role": "user", "content": request.question}]
    collected_sources: list[Source] = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_LEGAL_DATABASE_TOOL],
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use" or block.name != "search_legal_database":
                    continue

                query = block.input.get("query", request.question)
                n_results = block.input.get("n_results", 5)

                try:
                    results = retrieve(query=query, n_results=n_results)
                except Exception as e:
                    results = []
                    print(f"Retrieval error: {e}")

                for r in results:
                    meta = r.get("metadata", {})
                    source = Source(
                        case_name=meta.get("case_name", "Unknown"),
                        court=meta.get("court", "Unknown"),
                        date_filed=meta.get("date_filed", ""),
                        source_url=meta.get("source_url", ""),
                    )
                    if not any(s.source_url == source.source_url for s in collected_sources):
                        collected_sources.append(source)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _build_tool_result_content(results),
                    }
                )

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            answer_text = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
            return AskResponse(answer=answer_text, sources=collected_sources)


@app.post("/ingest")
async def ingest(request: IngestRequest) -> StreamingResponse:
    """Stream ingestion progress as Server-Sent Events."""
    global _ingest_running

    if _ingest_running:
        raise HTTPException(status_code=409, detail="Ingestion already in progress")

    queries = request.queries or SEED_QUERIES

    async def event_stream():
        global _ingest_running
        async with _ingest_lock:
            _ingest_running = True
            try:
                async for event in run_ingestion(queries):
                    yield f"data: {json.dumps(event)}\n\n"
            finally:
                _ingest_running = False
        yield "data: {\"event\": \"done\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/ingest/status")
async def ingest_status() -> dict:
    return {"running": _ingest_running}


@app.get("/stats")
async def stats() -> dict:
    return get_db_stats()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
