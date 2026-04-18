# FindLawByClaude — Claude Code Guide

## Project Context
Hackathon project built at Carnegie Mellon University (Claude Builder Club x CMUAI Hackathon, Apr 18 2026).

**Theme: Creative Flourishing** — Inspired by Dario Amodei's "Machines of Loving Grace". Build something that amplifies human creativity, helps people find meaning, or supports cultural expression and preservation.

## Project Goal
FindLawByClaude — help users discover, understand, and interact with legal information in a way that is accessible, meaningful, and empowering. (Update this as the project vision crystallizes.)

## Development Conventions
- No unnecessary comments; let code speak for itself
- Keep responses terse and direct
- Prefer editing existing files over creating new ones

## Stack
- Python 3.11+
- OpenRouter API (OpenAI-compatible) → `anthropic/claude-haiku-4-5`
- ChromaDB (local vector DB at ./chroma_db)
- sentence-transformers (all-MiniLM-L6-v2 embeddings)
- FastAPI + uvicorn
- openai Python SDK (pointed at OpenRouter)

## Running
1. `cp .env.example .env` → fill in `OPENROUTER_API_KEY`
2. `pip install -r requirements.txt`
3. `python src/ingest.py`        # populate the vector DB (shows tqdm progress)
4. `uvicorn src.api:app --reload` # start API on :8000
5. `open frontend/index.html`    # or serve with any static file server

## API Endpoints
- `POST /ask` — `{question}` → `{answer, sources[]}`
- `POST /ingest` — SSE stream of ingestion progress events
- `GET  /ingest/status` — `{running: bool}`
- `GET  /stats` — `{total_chunks: int}`
- `GET  /health`

## Key Files
- `src/ingest.py` — CourtListener fetch + ChromaDB upsert + CLI with tqdm
- `src/rag.py` — embedding + retrieval + OpenAI tool spec
- `src/api.py` — FastAPI routes, OpenRouter tool-use loop
- `frontend/index.html` — chat UI + live ingest panel
