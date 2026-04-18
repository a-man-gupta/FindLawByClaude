import asyncio
import json
import os
import re

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
MODEL = "anthropic/claude-haiku-4-5"          # fast chat
MODEL_DEEP = "anthropic/claude-opus-4-7"       # deep IRAC analysis with extended thinking
MAX_TOKENS = 4096
THINKING_BUDGET = 2500

SYSTEM_PROMPT = (
    "You are FindLawByClaude, an AI legal research assistant. "
    "You have access to a database of real US court opinions. "
    "When answering legal questions, always search the database first to ground your answer in actual case law. "
    "Cite specific cases by name when they are relevant. "
    "Format your response with clear sections using markdown: bold key terms, use bullet points for lists of factors or elements. "
    "Be clear that you are not providing legal advice and users should consult a licensed attorney for their specific situation."
)

SYSTEM_PROMPT_DEEP = (
    "You are FindLawByClaude in Deep Analysis mode — a senior legal research assistant. "
    "You have been provided with relevant case law retrieved from a database of US court opinions.\n\n"
    "Your response MUST follow this exact two-part structure:\n\n"
    "PART 1 — Reasoning (inside <thinking>...</thinking> tags):\n"
    "Walk through your analysis step-by-step BEFORE writing the final answer. "
    "Note the controlling doctrine, weigh how each retrieved case applies (or doesn't), "
    "consider counterarguments, surface ambiguities, and explain WHY you reach each conclusion. "
    "Be candid about uncertainty. This block should be substantive — at least 200 words.\n\n"
    "PART 2 — Final IRAC analysis (after </thinking>):\n"
    "Use markdown headers in this exact order:\n"
    "  ## Issue — what legal question is at stake?\n"
    "  ## Rule — what binding precedent applies? Cite cases by name.\n"
    "  ## Application — how does the rule map to the fact pattern? Address evidence cutting both ways.\n"
    "  ## Conclusion — most likely legal outcome and confidence level.\n\n"
    "End with a one-line disclaimer that this is not legal advice. "
    "Always include both parts in every response."
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


class DeepRequest(BaseModel):
    question: str


class DeepResponse(BaseModel):
    answer: str
    thinking: str
    sources: list[Source]
    model: str


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


@app.post("/deep", response_model=DeepResponse)
async def deep(request: DeepRequest) -> DeepResponse:
    """Deep IRAC analysis with Opus 4.7 + extended thinking.

    Directly retrieves top-K relevant cases (no agentic loop), hands them to
    Opus 4.7 with reasoning enabled, and returns answer + visible thinking.
    """
    client = _get_client()

    try:
        # Over-fetch chunks, then dedupe to ~8 unique cases for broader IRAC coverage.
        raw_results = retrieve(query=request.question, n_results=24)
    except Exception as e:
        print(f"Retrieval error in /deep: {e}")
        raw_results = []

    seen_opinions: set[str] = set()
    sources: list[Source] = []
    excerpts: list[str] = []
    for r in raw_results:
        meta = r.get("metadata", {}) or {}
        opinion_id = str(meta.get("opinion_id") or meta.get("source_url") or meta.get("case_name") or "")
        if not opinion_id or opinion_id in seen_opinions:
            continue
        seen_opinions.add(opinion_id)

        source = Source(
            case_name=meta.get("case_name", "Unknown"),
            court=meta.get("court", "Unknown"),
            date_filed=meta.get("date_filed", ""),
            source_url=meta.get("source_url", ""),
        )
        sources.append(source)
        excerpts.append(
            f"### {source.case_name} ({source.court}, {source.date_filed})\n"
            f"{r['text'][:1500]}"
        )
        if len(sources) >= 8:
            break

    case_context = (
        "\n\n".join(excerpts)
        if excerpts
        else "(No directly relevant cases found in the database.)"
    )

    user_content = (
        f"**Question:** {request.question}\n\n"
        f"**Relevant case law from the database:**\n\n{case_context}\n\n"
        f"Now provide your IRAC analysis."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_DEEP,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_DEEP},
                {"role": "user", "content": user_content},
            ],
            extra_body={"reasoning": {"max_tokens": THINKING_BUDGET}},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Opus call failed: {e}") from e

    msg = response.choices[0].message
    raw_text = msg.content or ""

    # Primary extraction: <thinking>...</thinking> tag in the response.
    # We instruct the model to emit this in SYSTEM_PROMPT_DEEP — works regardless
    # of provider-specific reasoning param support.
    thinking_text = ""
    answer_text = raw_text
    tag_match = re.search(
        r"<thinking>\s*(.*?)\s*</thinking>", raw_text, re.DOTALL | re.IGNORECASE
    )
    if tag_match:
        thinking_text = tag_match.group(1).strip()
        answer_text = re.sub(
            r"<thinking>.*?</thinking>\s*",
            "",
            raw_text,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    # Fallback: if no tag, check provider reasoning fields (some models populate these)
    if not thinking_text:
        reasoning_attr = getattr(msg, "reasoning", None)
        if isinstance(reasoning_attr, str) and reasoning_attr.strip():
            thinking_text = reasoning_attr
        else:
            details = getattr(msg, "reasoning_details", None)
            if details:
                parts = []
                for block in details:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("data") or ""
                    elif hasattr(block, "text"):
                        text = getattr(block, "text", "") or ""
                    else:
                        text = ""
                    if text:
                        parts.append(str(text))
                thinking_text = "\n\n".join(parts)

    return DeepResponse(
        answer=answer_text,
        thinking=thinking_text,
        sources=sources,
        model=MODEL_DEEP,
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
