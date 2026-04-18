# FindLawByClaude — Claude Code Guide

## Project Context
Hackathon project built at Carnegie Mellon University (Claude Builder Club x CMUAI Hackathon, Apr 18 2026).

**Theme: Creative Flourishing** — Inspired by Dario Amodei's "Machines of Loving Grace". Build something that amplifies human creativity, helps people find meaning, or supports cultural expression and preservation.

## Project Goal
FindLawByClaude — help users discover, understand, and interact with legal information in a way that is accessible, meaningful, and empowering. (Update this as the project vision crystallizes.)

## Development Conventions
- Language/stack: TBD — update this section when decided
- No unnecessary comments; let code speak for itself
- Keep responses terse and direct
- Prefer editing existing files over creating new ones

## Key Commands
- Add build/test/run commands here as the project grows

## Stack
- Python 3.11+
- ChromaDB (local vector DB at ./chroma_db)
- sentence-transformers (all-MiniLM-L6-v2 embeddings)
- Anthropic SDK (claude-sonnet-4-6 with tool use)
- FastAPI + uvicorn

## Running
1. cp .env.example .env && fill in ANTHROPIC_API_KEY
2. pip install -r requirements.txt
3. python src/ingest.py   # populate the vector DB
4. uvicorn src.api:app --reload  # start API
5. open frontend/index.html

## Claude API Usage
- Use `claude-sonnet-4-6` (current model) or `claude-opus-4-7` for complex tasks
- Enable prompt caching wherever possible
- See: Anthropic SDK docs for tool use, streaming, and batch patterns
