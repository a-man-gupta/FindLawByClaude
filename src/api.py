import asyncio
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

from .ingest import SEED_QUERIES, get_db_stats, run_ingestion
from .rag import OPENAI_TOOL_SPEC, retrieve

load_dotenv()

app = FastAPI(title="FindLawByClaude")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-haiku-4-5"  # OpenRouter model ID
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


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def _build_tool_result(results: list[dict]) -> str:
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
    client = _get_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.question},
    ]
    collected_sources: list[Source] = []

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            tools=[OPENAI_TOOL_SPEC],
            tool_choice="auto",
            messages=messages,
        )

        choice = response.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.function.name != "search_legal_database":
                    continue

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                query = args.get("query", request.question)
                n_results = args.get("n_results", 5)

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

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": _build_tool_result(results),
                    }
                )
        else:
            return AskResponse(
                answer=msg.content or "",
                sources=collected_sources,
            )


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
