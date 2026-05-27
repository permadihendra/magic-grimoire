# Magic Grimoire 📖✨ — RAG Study Agent

> A Telegram bot powered by LlamaIndex + local Ollama models that ingests your study documents
> (PDFs, EPUBs, notes) and lets you query them conversationally. Like having a personal study
> assistant that generates quizzes, summaries, and exam prep from YOUR materials.

**Stack:** Python 3.11+ · FastAPI · LlamaIndex · Ollama (magic-grimoire:3b) · nomic-embed-text · aiosqlite → sqlite3 · python-telegram-bot · Gemini 3.1 Flash Lite (agent brain)

---

## Implementation Status

### ✅ Phase 1: Foundation — COMPLETE
1. ✅ Project scaffolded, old plugins stripped
2. ✅ pyproject.toml — LlamaIndex + Ollama deps
3. ✅ .env.example — Ollama-focused
4. ✅ config.py — pydantic-settings
5. ✅ database.py — sqlite3 schema (documents + query_history + knowledge_pairs)
6. ✅ main.py — lifespan + plugin loading
7. ✅ dispatcher.py — BrainPlugin routing
8. ✅ gateway.py — Telegram webhook receiver
9. ✅ start-bot.sh — Quick tunnel launcher with Ollama checks

### ✅ Phase 2: RAG Core — COMPLETE
10. ✅ app/rag/models.py — Ollama LLM + embedding config
11. ✅ app/rag/indexer.py — Document ingestion (LiteParse + SimpleDirectoryReader fallback)
12. ✅ app/rag/engine.py — RAG query engine (qa/quiz/summary/sources)
13. ✅ app/rag/prompts.py — Study prompt templates
14. ✅ app/rag/guard.py — OllamaGuard: semaphore, health check, crash recovery
15. ✅ app/rag/feedback.py — FeedbackLearner: per-session penalty/boost

### ✅ Phase 3: Agentic Brain — COMPLETE
16. ✅ Gemini agent (app/llm/gemini.py) — intent classification + tool routing
17. ✅ BrainPlugin — Gemini decides tools: ask, quiz, summarize, list_docs, chat, retrieve, feedback
18. ✅ Post-retrieval filtering — guaranteed named-doc chunks in context
19. ✅ Document-aware score boosting — 2x for named documents

### ✅ Phase 4: UX & Progress — COMPLETE
20. ✅ Model warm-up on startup (~2s for 3b)
21. ✅ ProgressWatcher — phase-aware async task, immediate report at t=0.5s
22. ✅ Knowledge cache (Q&A pairs with embedding search) — 0.1s vs 10-30s
23. ✅ Message truncation — Telegram 4096 char limit safety
24. ✅ No-split-queries rule — agent prompt prevents two ask() calls

### ✅ Phase 5: Stability & Guardrails — COMPLETE
25. ✅ Ollama semaphore — only 1 concurrent operation (prevents VRAM crash)
26. ✅ Pre-call health check — detects dead Ollama
27. ✅ Per-call timeout — 120s generation, 60s warm-up, 600s indexing
28. ✅ Embedding batch limit — 50 docs per batch
29. ✅ Startup health verification
30. ✅ Fresh sqlite3 per operation + WAL checkpoint — prevents GPU-load corruption

### ✅ Phase 6: Retrieval Quality — COMPLETE
31. ✅ Group-sort — doc_nodes first, sorted within groups
32. ✅ Primary/Also source labels — clear citation hierarchy
33. ✅ Feedback learning — per-chat penalty/boost from user feedback
34. ✅ Document name matching — _filename_match(), normalizes separators

### ✅ Phase 7: Cleanup & Docs — COMPLETE (this phase)
35. ✅ Consolidated PLAN.md
36. ✅ README.md updated
37. ✅ All improvement plans merged

### 🔮 Future Improvements

| Plan | Description | Priority |
|---|---|---|
| **Streaming output** | Token-by-token LLM output to Telegram | 🟡 High |
| **Full agentic loop** | Multi-turn Gemini with tool result feedback | 🟡 High |
| **Gemini filename cleaning** | Extract book titles at index time | ✅ Done |
| **Hybrid search** | BM25 + vector for keyword matching | 🟢 Medium |
| **LLM reranker** | Small model reranks retrieved chunks | 🟢 Medium |

---

## Architecture

```
Telegram User
  ↓
cloudflared tunnel → FastAPI (gateway.py)
  ↓
Dispatcher → no slash? → BrainPlugin (Gemini agent)
  ↓
Gemini decides tool → ask / quiz / summarize / list_docs / chat / feedback
  ↓
┌─ Knowledge Cache (sqlite3) ────────────────────────┐
│  Embed query → cosine similarity against stored    │
│  Q&A pairs → if match > 0.92, return cached answer │
│  ~150ms vs 10-30s                                  │
└────────────────────────────────────────────────────┘
  ↓ (cache miss)
RAG Engine (LlamaIndex)
  ├─ Retrieve (sync) — top-k chunks
  ├─ Document boosting + feedback adjustments
  ├─ Post-retrieval filtering (force named-doc chunks)
  ├─ Group-sort (doc_nodes first)
  └─ LLM generation (magic-grimoire:3b, temp 0.0)
  ↓
Source citations (display_name) → Telegram response
```

### VRAM Budget (8 GB RTX 3050)

| Component | Usage |
|---|---|
| magic-grimoire:3b (weights) | 1.9 GB |
| KV cache (4096 ctx) | ~0.5 GB |
| nomic-embed-text | ~0.3 GB |
| CUDA overhead | ~0.5 GB |
| **Total** | **~3.2 GB — 4.8 GB headroom** |

---

## Components

### BrainPlugin — Agentic Router

The `brain` plugin receives ALL free text messages. It sends them to Gemini which decides:
- Which tool to call
- Whether to use knowledge cache
- When to chain tools (feedback → retry)

### StudyPlugin — RAG Backend

Exposes methods called by BrainPlugin tools:
- `ask_query(query, difficulty, document, chat_id)` → RAG answer + sources
- `retrieve_passages(query, document, chat_id)` → raw passages (no LLM)
- `_handle_index()` → rebuild document index

### RAG Engine

- Uses sync retrieval (`retriever.retrieve()`) — async version hangs
- Post-retrieval filtering guarantees named-doc chunks in context
- Group-sort keeps doc_nodes first, prevents wrong source citation
- All Ollama calls go through `OllamaGuard` semaphore

### Knowledge Cache

- Q&A pairs stored in sqlite3 with embeddings as BLOBs
- Cosine similarity + source document matching
- Cache check runs BEFORE retrieval (~150ms)
- Auto-store on successful generation
- Invalidated on `/index`

### Guardrails

| Guardrail | Mechanism |
|---|---|
| Concurrent Ollama | `asyncio.Semaphore(1)` |
| Dead Ollama | `ollama_is_alive()` before every call |
| Timeout | `asyncio.wait_for()` per operation |
| DB corruption | Fresh sqlite3 per op + WAL checkpoint |
| Telegram overflow | `_safe_truncate()` to 4000 chars |
| Progress watcher | `ProgressWatcher` at t=0.5s, 30s, 60s |
| Rate limits | Embedding batches of 50 |

---

## Repository Layout

```
magic-grimore/
├── app/
│   ├── main.py               # FastAPI app + lifespan
│   ├── config.py             # pydantic-settings
│   ├── database.py           # sqlite3 — documents + knowledge_pairs + query_history
│   ├── bot/
│   │   ├── gateway.py        # Telegram webhook route
│   │   ├── dispatcher.py     # Route free text → BrainPlugin
│   │   ├── middlewares.py    # Rate limiter + auth
│   │   ├── context.py        # BotContext dataclass
│   │   └── setup_webhook.py  # One-time webhook registration
│   ├── plugins/
│   │   ├── base.py           # Plugin ABC + registry
│   │   ├── brain/
│   │   │   └── handler.py    # Gemini agent + tool router
│   │   ├── study/
│   │   │   └── handler.py    # RAG study backend
│   │   └── system/
│   │       └── handler.py    # /start, /help, /ping
│   ├── rag/
│   │   ├── models.py         # Ollama LLM + embedding config
│   │   ├── indexer.py        # Document ingestion
│   │   ├── engine.py         # RAG query engine
│   │   ├── prompts.py        # Study prompt templates
│   │   ├── guard.py          # OllamaGuard semaphore + health
│   │   ├── feedback.py       # FeedbackLearner
│   │   └── knowledge_cache.py # Q&A cache
│   ├── ui/
│   │   └── progress.py       # ProgressWatcher
│   └── llm/
│       └── gemini.py         # Gemini API client
├── data/
│   ├── magic-grimoire.db     # sqlite3 database
│   ├── index_storage/        # LlamaIndex vector store
│   └── cloudflared.log
├── docs/                     # Drop your PDFs/EPUBs here!
├── Modelfile.3b              # Optimized qwen2.5:3b config
├── Modelfile.qwen-stable     # Optimized qwen2.5:7b config
├── start-bot.sh              # Quick tunnel launcher
├── pyproject.toml
└── .env.example
```

---

## Database Schema

```sql
documents:
  id, filename, filepath, file_size, chunk_count, display_name, indexed_at

query_history:
  id, chat_id, query, response, created_at

knowledge_pairs:
  id, question, question_embedding (BLOB), short_answer, full_answer,
  source_doc, retrieval_score, approved, used_count, chat_id, created_at, last_used_at
```

---

## Deployment

```bash
# Prerequisites
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama create magic-grimoire:3b -f Modelfile.3b

# Install
git clone <repo> magic-grimoire
cd magic-grimoire
cp .env.example .env && chmod 600 .env
uv sync

# Start
./start-bot.sh
```

---

## Model Optimization Parameters

The custom `magic-grimoire:3b` model (from `Modelfile.3b`):

| Parameter | Value | Reason |
|---|---|---|
| `num_ctx` | 4096 | KV cache ~0.5 GB (fits 8 GB card) |
| `temperature` | 0.0 | Deterministic — facts from documents, not randomness |
| `num_predict` | 1024 | Concise study answers |
| `repeat_penalty` | 1.05 | Light loop prevention |
| `num_gpu` | 99 | Force all layers to GPU |
| `top_k` | 1 | Irrelevant with temp 0.0 |
| `top_p` | 1.0 | Irrelevant with temp 0.0 |
