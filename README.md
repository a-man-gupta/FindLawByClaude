# ⚖️ FindLawByClaude

> AI-powered legal research assistant grounded in real US Supreme Court case law. Ask in plain English, get answers backed by actual precedent — never invented citations.

**Built at:** Carnegie Mellon University · *Claude Builder Club × CMUAI Hackathon* · April 2026
**Theme:** Creative Flourishing — inspired by [Dario Amodei's "Machines of Loving Grace"](https://www.anthropic.com/news/machines-of-loving-grace)

---

## The Problem

Legal information is gatekept. Westlaw and LexisNexis cost ~$300/month. Most people who need to understand their rights — tenants facing eviction, creators worried about fair use, employees questioning a contract — can't afford an attorney just to read them a case. ChatGPT will happily invent fake citations. Google returns SEO blogspam.

**The 80% of Americans without legal counsel deserve direct access to real precedent.**

## What It Does

FindLawByClaude is a two-tier RAG legal assistant:

- **`/ask` — Fast tier (Claude Haiku 4.5):** Answers in ~3 seconds with cited cases retrieved from a vector database of real US Supreme Court opinions. Click any citation → opens the actual case on Oyez or CourtListener.

- **`/deep` — Deep tier (Claude Opus 4.7):** Click "🧠 Deep Analysis" on any answer to get a senior-partner-quality IRAC analysis with **visible chain-of-thought reasoning**. Watch Claude weigh precedents in real time before producing the final analysis.

- **Jurisdictional honesty:** State-law questions (eviction, family, employment) trigger a clarifying question instead of a hallucinated holding. The system prompt explicitly forbids inventing case citations.

## Demo

```
You: Is burning the American flag protected speech under the First Amendment?

FindLawByClaude (3 sec):
The Supreme Court has consistently held that flag burning is protected
expressive conduct under the First Amendment...

📚 Sources cited:
  • United States v. Eichman (U.S. Supreme Court, 1990)  → View source
  • Texas v. Johnson (U.S. Supreme Court, 1989)          → View source
  • Watts v. United States (U.S. Supreme Court, 1969)    → View source
  ... 9 more

[🧠 Deep Analysis (Opus 4.7)] ← click to see Claude's reasoning live
```

## Architecture

```
   User question
        │
        ├──► POST /ask  ──► Haiku 4.5 + tool-use loop
        │                   ├─► search_legal_database tool
        │                   │    └─► ChromaDB (cosine, MiniLM-L6-v2)
        │                   │         └─► Oyez SCOTUS corpus + CourtListener
        │                   └─► answer + cited sources
        │
        └──► POST /deep ──► Opus 4.7 + extended thinking
                            ├─► top-K retrieval (deduped by case)
                            ├─► structured <thinking>...</thinking> response
                            └─► thinking + IRAC + sources
```

**Backend:** FastAPI on `127.0.0.1:8000`
**Frontend:** Static HTML/CSS/JS served from `frontend/`
**LLM access:** OpenRouter (single key, both Haiku and Opus)
**Embeddings:** sentence-transformers `all-MiniLM-L6-v2` (CPU, local)
**Vector DB:** ChromaDB persistent on disk

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Then edit .env and set: OPENROUTER_API_KEY=sk-or-v1-...

# 3. Populate the database (~5 min for default Oyez corpus)
PYTHONIOENCODING=utf-8 python src/ingest.py

#    Optional: also pull ~30k SCOTUS cases from Harvard CAP
# PYTHONIOENCODING=utf-8 python src/ingest.py --scotus --max 2000

# 4. Run the API server
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000

# 5. Serve the frontend (separate terminal)
cd frontend && python -m http.server 8080
# Open http://localhost:8080
```

## API

| Endpoint | Method | Purpose | Latency |
|---|---|---|---|
| `/ask` | POST | Fast chat with Haiku 4.5 + tool use | ~3s |
| `/deep` | POST | IRAC analysis with Opus 4.7 + visible thinking | ~25s |
| `/ingest` | POST | SSE stream of ingestion progress | minutes |
| `/stats` | GET | Total chunks in DB | <1s |
| `/health` | GET | Liveness check | <1s |

**Request shape (both `/ask` and `/deep`):**
```json
{ "question": "Is burning the American flag protected speech?" }
```

**`/deep` response:**
```json
{
  "answer": "## Issue\n... ## Rule\n... ## Application\n... ## Conclusion\n...",
  "thinking": "Let me work through the controlling doctrine here. The Spence test for symbolic speech...",
  "sources": [{ "case_name": "...", "court": "...", "date_filed": "...", "source_url": "..." }],
  "model": "anthropic/claude-opus-4-7"
}
```

## Tech Stack

- **Models:** Claude Haiku 4.5 (chat) · Claude Opus 4.7 (deep reasoning)
- **LLM gateway:** OpenRouter (`openai` Python SDK pointed at `openrouter.ai/api/v1`)
- **Backend:** FastAPI + uvicorn
- **RAG:** ChromaDB + sentence-transformers MiniLM-L6-v2
- **Data:** Oyez API (SCOTUS landmarks) · CourtListener · Harvard Caselaw Access Project (optional)
- **Frontend:** Vanilla HTML/CSS/JS + marked.js (markdown rendering)
- **Languages:** Python 3.11+, ES6 JavaScript

## Why This Fits "Creative Flourishing"

Dario's essay calls out civic empowerment as a precondition for human flourishing — *"a beacon of hope that helps make liberal democracy the form of government that the whole world wants to adopt."* Legal literacy is one of the deepest forms of civic empowerment. When people can actually read the cases that govern their lives — not paywalled summaries, not ChatGPT hallucinations — they gain agency over their own rights.

This isn't replacing lawyers. It's giving the 80% of people who can't afford one a real first step.

## Honesty Guarantees

The system prompt enforces three rules:
1. **No fabricated citations.** If a case isn't in the retrieved results, the model says so explicitly.
2. **Jurisdictional caveats.** State-law questions (eviction, family, employment, contracts) trigger a clarifying question about the user's state instead of a guessed holding.
3. **Always disclaim.** Every response ends with "this is not legal advice — consult a licensed attorney."

## Project Layout

```
.
├── src/
│   ├── api.py        # FastAPI routes (/ask, /deep, /ingest, /stats, /health)
│   ├── rag.py        # Vector retrieval + tool spec for Claude
│   └── ingest.py     # Oyez + HuggingFace + CourtListener + Harvard CAP loaders
├── frontend/
│   └── index.html    # Chat UI + Deep Analysis button + thinking panel
├── requirements.txt
├── .env.example
└── CLAUDE.md         # Internal dev guide for working with the codebase
```

## Disclaimer

This is a **research and education tool**, not a substitute for licensed legal counsel. Every response includes a non-legal-advice disclaimer. Citations are accurate only to the extent the retrieved sources are accurate; always verify a case before relying on it.
