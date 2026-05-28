# Magic Grimoire — Agent Guide

> **Purpose:** This is the canonical reference for any coding agent working on this project.
> Read this file first. It tells you how everything works, what's implemented, and where to find things.

## Project Overview

**What it does:** Telegram bot + RAG study agent. Ingest PDFs/EPUBs → query them conversationally,
get answers with source citations, generate quizzes, summarize topics.

**Stack:** Python 3.11+ · FastAPI · LlamaIndex · Ollama (local LLM) · sentence-transformers (CPU fallback embeddings) · Gemini 3.1 Flash Lite (agent brain) · sqlite3 (fresh per-op) · python-telegram-bot

**Run:** `bash app/start-bot.sh` — always, never `uvicorn` directly. This script manages cloudflared tunnel.

---

## Repository Layout

```
app/
├── main.py                  # FastAPI app factory + lifespan
├── config.py                # pydantic-settings (all secrets here)
├── database.py              # sqlite3 (fresh connection per op, WAL mode)
│                           # Tables: documents, query_history, knowledge_pairs
│
├── bot/
│   ├── gateway.py           # Telegram webhook receiver (POST /telegram)
│   ├── dispatcher.py        # Slash commands → Plugin.handle(); free text → BrainPlugin
│   ├── middlewares.py       # Rate limiter + auth guard
│   ├── context.py           # BotContext dataclass builder
│   └── setup_webhook.py     # One-time webhook registration
│
├── plugins/
│   ├── base.py              # Plugin ABC: name, commands, description, handle()
│   │
│   ├── brain/handler.py     # BrainPlugin — Gemini agent (agentic router)
│   │   │                   #   Tools: ask, quiz, summarize, list_docs, chat,
│   │   │                   #          feedback, retrieve, delete
│   │   │                   #   Intercepts all free-text messages
│   │   └── __init__.py
│   │
│   ├── study/handler.py      # StudyPlugin — RAG backend (exposed via /ask, /quiz, /docs, /index, /files, /delete)
│   │   │                   #   Functions used by BrainPlugin tools:
│   │   │                   #   ask_query(), generate_quiz(), summarize_topic(),
│   │   │                   #   retrieve_passages(), _tool_list_docs()
│   │   └── __init__.py
│   │
│   └── system/handler.py     # SystemPlugin (/start, /help, /ping, /status)
│
├── rag/
│   ├── models.py            # Ollama LLM + FallbackEmbedding (Ollama→sentence-transformers)
│   ├── indexer.py           # Multi-method document parsing + embedding + indexing
│   │   │                   #   Parsers: ebooklib > LiteParse > Calibre CLI > SimpleDirectoryReader
│   │   │                   #   Key: parse_document_multimethod(), probe_index(), _sync_documents()
│   ├── engine.py            # RAG engine: retrieval + TreeSummarize synthesis
│   │   │                   #   Key: query_with_sources() → returns answer + sources + diagnostics
│   ├── guard.py             # OllamaGuard (semaphore + VRAM check + OOM recovery)
│   │   │                   #   Key: ollama_check_vram(), OllamaGuard.__aenter__()
│   ├── feedback.py          # FeedbackLearner (per-chat document penalty/boost)
│   ├── knowledge_cache.py   # Q&A pair cache with embedding cosine search
│   ├── embeddings.py        # FallbackEmbedding (tries Ollama → auto-switches to CPU all-MiniLM-L6-v2)
│   └── prompts.py           # QA / Quiz / Summary prompt templates (difficulty variants)
│
├── ui/
│   ├── progress.py           # ProgressState + ProgressWatcher (phase-aware, escalating)
│   └── helpers.py            # _build_file_card(), _build_files_card(), _build_total_footer()
│                           # _simple_clean_filename(), _fmt_timestamp()
│
└── llm/
    └── gemini.py            # Gemini API client (gemini_chat())
```

---

## Architecture: Message Flow

```
Telegram message
  ↓
gateway.py (POST /telegram) → parse update
  ↓
dispatcher.py (dispatch update)
  ├─ Slash command → Plugin.handle() by command
  └─ Free text    → BrainPlugin.handle() [Gemini agent]
        ↓
        Gemini classifies intent + decides tools
        ↓
        ┌─ ask()   → study/handler.ask_query() → engine.query_with_sources()
        ├─ quiz()  → study/handler.generate_quiz()
        ├─ summarize() → study/handler.summarize_topic()
        ├─ list_docs() → brain/handler._tool_list_docs_v2()
        ├─ feedback() → rag/feedback.py FeedbackLearner
        ├─ retrieve() → rag/engine.retrieve_only()
        └─ chat()   → Ollama direct (no RAG)
              ↓
        Response edited into Telegram "thinking" message
```

---

## Key Files

### `app/plugins/brain/handler.py` — Agent Brain
- **`AGENT_PROMPT`** (system prompt at top): Tells Gemini how to route intents, tool syntax, rules
- **`_tool_ask()`** — calls `ask_query()` → `query_with_sources()` → returns answer with citations
- **`_tool_quiz()`** — calls `generate_quiz()` with source info + next-step + stats footer
- **`_tool_summarize()`** — calls `summarize_topic()` with source info + next-step + stats footer
- **`_tool_list_docs_v2()`** — lists documents with word counts, parse method, probe status
- **`_tool_feedback()`** — records user feedback in FeedbackLearner per chat
- **`_tool_retrieve()`** — fast retrieval pass (no LLM), returns raw passages
- **`_tool_chat()`** — casual Ollama chat (no RAG)
- **`_conversation_memory`** — last 5 exchanges per chat for context
- **`_TOOL_REGISTRY`** dict at bottom — maps tool names to handler functions
- **`_SLOW_TOOLS = {"ask", "quiz", "summarize"}`** — trigger thinking indicator
- If no `TOOL:` lines in Gemini response → appends "ℹ️ This answer came from my training data" footer

### `app/plugins/study/handler.py` — RAG Backend
- **`ask_query()`** — main RAG query: ensure_index → query_with_sources → returns answer str
- **`generate_quiz()`** — retrieve_only (for sources) → query(mode="quiz") → append sources + next-step + stats
- **`summarize_topic()`** — same pattern as generate_quiz but mode="summary"
- **`_handle_index()`** — 3-phase: pre-check (test-parse each file) → build index → post-build probe + per-file report
- **`_handle_docs()`** — delegates to `_tool_list_docs_v2()`
- **`_handle_files()`** — lists files on disk with indexed status + metadata from DB
- **`_handle_delete()`** — deletes file, shows impact (word/chunk count from DB before deleting)
- **`_doc_indexer`** — module-level `DocumentIndexer` singleton
- **`_rag_engine`** — module-level `RAGEngine` singleton

### `app/rag/engine.py` — Query Engine
- **`RAGEngine.query_with_sources()`** — the main method:
  1. Check index health (L1 guardrail)
  2. Check knowledge cache (L0 fast path)
  3. Sync retrieval + document boosting + feedback adjustment
  4. Post-retrieval filtering: force named-doc chunks into context
  5. L1b: 0 chunks → early return (no LLM call)
  6. Token budget check → early return if insufficient headroom
  7. L3-L6: OllamaGuard → TreeSummarize (30s hard timeout → fallback to simple prompt)
  8. L7: Validate answer length → generic "no answer" if too short
  9. Build source citations with score labels
  10. Add next-step suggestion 💡
  11. Add diagnostics footer 📊
  12. Store in knowledge cache (async)
  13. Return `{answer: str, sources: list[dict]}`
- **`_lookup_display_name()`** — maps basename → display_name from DB (memoized)
- **`retrieve_only()`** — raw retrieval without LLM (used by quiz/summarize pre-check)
- **`query()`** — legacy simple query (no sources, used by quiz/summary mode)

### `app/rag/indexer.py` — Document Indexer
- **`parse_document_multimethod()`** — tries parsers in order: ebooklib → LiteParse → Calibre → SimpleDirectoryReader
  Returns `ParseResult(success, method, word_count, char_count, doc)`
- **`probe_index()`** — test retrieval to verify index is actually usable
- **`DocumentIndexer.rebuild_index(report_fn)`** — full rebuild with `report_fn(chunk_count)`
- **`DocumentIndexer.pre_index_check_single()`** — test-parse one file before building
- **`_sync_documents()`** — writes to `documents` table: filename, filepath, file_size, display_name, word_count, chunk_count, parse_method, verified

### `app/rag/models.py` — Model Config
- **`get_llm()`** — Ollama LLM (magic-grimoire:3b), temperature=0.0, num_predict=128, num_ctx=1024
- **`get_embed_model()`** — FallbackEmbedding(primary=OllamaEmbedding(embed_batch_size=3))
  - Tries Ollama nomic-embed-text first
  - On failure → auto-switches to CPU sentence-transformers/all-MiniLM-L6-v2
- **`warm_up()`** — sends 2-token request to pre-load LLM into VRAM on startup
- **`configure_settings()`** — sets LlamaIndex global Settings

### `app/rag/guard.py` — Ollama Guard
- **`ollama_check_vram()`** — calls nvidia-smi, returns (free_mb, used_mb, total_mb)
- **`OllamaGuard.__aenter__()`** — checks VRAM + acquires semaphore
  - Blocks if >1 concurrent Ollama operation
  - Returns guard object; raises if <500MB free VRAM or Ollama dead
- **`OllamaBusyError`, `OllamaDeadError`** — raised instead of generic exceptions
- **`verify_ollama_on_startup()`** — called in main.py lifespan, logs warning but doesn't crash

### `app/rag/embeddings.py` — Fallback Embedding
- **`FallbackEmbedding`** — wraps OllamaEmbedding + sentence-transformers
  - `_async_get_text_embedding()` — tries primary, falls back on error
  - `_async_get_text_embedding_batch()` — tries primary, falls back on error
  - `_aget_query_embedding()`, `_get_query_embedding()` — for retrieval-time embedding
  - `_using_fallback` property — true once sentence-transformers activated

### `app/rag/knowledge_cache.py` — Q&A Cache
- **`search_cache()`** — cosine similarity on stored question embeddings
  - Returns cached answer if similarity ≥ 0.92
  - Fast path: bypasses RAG entirely for repeated queries
- **`store_pair()`** — stores (question, question_embedding, answer, source_doc) in `knowledge_pairs`

### `app/rag/feedback.py` — Feedback Learning
- **`FeedbackLearner`** — per-chat document scoring
  - `penalize_document()` — reduces score for wrongly-cited docs
  - `boost_document()` — increases score for correctly-cited docs
  - `get_blocked_docs()` — returns set of docs to deprioritize
  - `adjust_score()` — called in engine after retrieval, before LLM

### `app/ui/progress.py` — Progress Watcher
- **`ProgressState`** — tracks phase, chunks_embedded, files_total, files_parsed, passges_found
- **`ProgressWatcher`** — asyncio task that sends Telegram edits at t=0.5s, then escalating intervals
  - Phases: index_parse → index_embed → generate → ...
  - Reports current state every `interval` seconds

### `app/ui/helpers.py` — UI Helpers
- **`_build_file_card()`** — builds a document card for `/docs` (filename, sizes, parse_method, verified, timestamp)
- **`_build_files_card()`** — builds a file card for `/files` (indexed/pending status + display_name)
- **`_build_total_footer()`** — summary line with total counts
- **`_simple_clean_filename()`** — strips cruft from raw filenames (regex patterns)
- **`_fmt_timestamp()`** — formats DB timestamp to "28 May 2026, 05:08"

---

## Database Schema

```sql
documents (
  id, filename, filepath UNIQUE, file_size,
  display_name,       -- Gemini-cleaned title
  word_count,         -- extracted from parsing
  chunk_count,        -- indexed chunks
  parse_method,       -- "ebooklib (EPUB)", "SimpleDirectoryReader", etc.
  verified,            -- 1 if probe_index() passed after last build
  indexed_at           -- last re-index timestamp
)

query_history (id, chat_id, query, response, created_at)

knowledge_pairs (
  id, question, question_embedding BLOB,
  short_answer, full_answer, source_doc,
  retrieval_score, approved, used_count, chat_id,
  created_at, last_used_at
)
```

---

## Commands

| Command | Handler | Output |
|---|---|---|
| `/ask <q>` | `StudyPlugin._handle_ask()` | Answer + source citations + next-step + diagnostics |
| `/quiz <topic>` | `StudyPlugin._handle_quiz()` | Questions + sources + next-step + retrieval stats |
| `/summarize <topic>` | `StudyPlugin._handle_summarize()` | Summary + sources + next-step + stats |
| `/qna <topic>` | `StudyPlugin._handle_qna()` | Comprehension Q&A pairs (default 10) |
| `/docs` | `StudyPlugin._handle_docs()` | Document library with word counts, parse methods, probe status |
| `/index` | `StudyPlugin._handle_index()` | Pre-check → parse → embed → probe → per-file report |
| `/files` | `StudyPlugin._handle_files()` | All files on disk with indexed status + metadata |
| `/delete <id>` | `StudyPlugin._handle_delete()` | Deletes file + shows impact (size/words/chunks) |
| Free text | `BrainPlugin.handle()` | Gemini routes to appropriate tool |

**Free text routing:** Gemini decides the tool, StudyPlugin is only invoked when `/ask` etc. are typed explicitly. For free text, BrainPlugin's `_tool_ask()` calls `ask_query()` directly.

---

## Response Format Pattern

All RAG responses follow this structure:
```
[Answer text...]

📖 Primary source: The Bhagavad Gita     ← if document was found
📎 Also: Chapter Notes.txt               ← additional sources

💡 Next steps: Would you like a follow-up question, a `quiz` on this topic,
   or a `summary` of the key points? Just ask!

📊 Retrieval: 3 chunks | ~420 tokens | k=3 | ctx=2048  ← diagnostics footer
```

Quiz/summary add `📖 Sources used:` between answer and next-step.

---

## Modelfile Parameters (magic-grimoire:3b)

```
temperature=0.0          # Deterministic — defense in depth
num_ctx=2048             # Context window (API can extend to 4096)
num_gpu=99               # All layers on GPU (power capped via MSI Afterburner)
num_predict=2048         # Max output tokens (sent via API, overrides Modelfile)
repeat_penalty=1.1       # Slight discourage of repetition
num_thread=4             # CPU thread count
```

**VRAM budget (RTX 3050 8GB, optimized):**
- LLM weights: 1.9 GB
- KV cache: ~2.4 GB (2048 ctx)
- Embeddings (nomic): 0.3 GB
- **Total: ~4.6 GB** (3.4 GB headroom)
- Power capped at 80% via MSI Afterburner (~100W)

---

## Guardrails (7 Layers)

| Layer | Location | Guards Against |
|---|---|---|
| L1 | `engine.query_with_sources()` | 0 chunks → early return, no LLM call |
| L1b | `engine.query_with_sources()` | Token budget <50 → early return |
| L2 | `engine.query_with_sources()` | 0 docs in index → early return |
| L3 | `Modelfile.3b` | num_ctx=1024, num_predict=128, temp=0.0 |
| L4 | `guard.py OllamaGuard` | VRAM pre-check before every Ollama call |
| L5 | `engine.query_with_sources()` | TreeSummarize 30s timeout → fallback prompt |
| L6 | `models.py warm_up()` | 2-token request pre-loads model into VRAM |
| L7 | `embeddings.py FallbackEmbedding` | Ollama embed crash → auto-switch to CPU |

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Fresh sqlite3 per operation | No background-thread corruption during GPU load |
| FallbackEmbedding | Ollama Go runner crashes during batch embed → use CPU all-MiniLM-L6-v2 |
| embed_batch_size=3 | Was 10 — too many concurrent calls crashed Ollama Go runtime |
| os.path.basename for file_name metadata | Compare reliably between os.listdir() vs stored metadata |
| tree_summarize over manual acomplete | Hierarchical synthesis but CPU/GPU intensive; 30s timeout + fallback protects |
| num_ctx=1024 over 2048 | Halves KV cache → more VRAM headroom |
| Temperature 0.0 | Deterministic output, defense in depth (Modelfile + models.py) |
| Gemini for agent routing | Fast (~50ms), free tier, better intent classification than Ollama |
| Detection footer for no-TOOL | If Gemini doesn't call a tool → "ℹ️ from training data" so user knows |

---

## Secret Safety Rules

| Rule | Enforcement |
|---|---|
| `.gitignore` first | Always verify before `git add` |
| `shell=True` forbidden | Always `create_subprocess_exec` with explicit arg list |
| No threading | `asyncio` only; never `import threading` |
| Parameterized SQL | All DB writes use `?` placeholders |
| Secrets via `settings` | Every credential from `settings.*` |
| No secrets in logs | No `logging.debug(settings.*)` |
| Explicit HTTP timeouts | Every `httpx` call sets `timeout=` |
| Use `logging` | No `print()` in production paths |
| `.env.example` fake values | e.g. `your-telegram-bot-token-here` |

---

## Common Commands

```bash
bash app/start-bot.sh            # Always use this — manages cloudflared tunnel
bash app/stop-bot.sh             # Kill all bot processes cleanly
bash app/dependency-check.sh     # Verify all deps (--fix to auto-install)
uv sync                          # Install deps
uv sync --extra epub             # Install EPUB support (ebooklib + html2text)
uv sync --extra dev              # Install dev deps
uv run ruff check .              # Lint
uv run python3 -c "..."          # Quick test

ollama list                      # Check models
ollama show magic-grimoire:3b    # Inspect model info
nvidia-smi                       # Check VRAM

# Register webhook (one-time)
uv run python -m app.bot.setup_webhook

# Remove webhook
uv run python -m app.bot.setup_webhook --delete
```

---

## Ollama Models

```bash
ollama create magic-grimoire:3b -f Modelfile.3b    # Optimized qwen2.5:3b
ollama pull nomic-embed-text                       # Embeddings
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Bot doesn't reply | Re-run `bash app/start-bot.sh` (cloudflared tunnel expired) |
| "Model not found" | `ollama create magic-grimoire:3b -f Modelfile.3b` |
| Index rebuild fails | `ollama serve` + `ollama list` to verify models |
| "Ollama embed Go runner crashed" | Bot auto-switches to CPU fallback — retry `/index` |
| "database disk image is malformed" | Delete `data/magic-grimoire.db` and restart |
| PC crashed during `/index` | Install `sentence-transformers` for CPU fallback |

---

## Related

- [PLAN.md](PLAN.md) — Development roadmap
- [PLAN_audit_commands.md](PLAN_audit_commands.md) — Command UI audit plan
- [PLAN_guardrails.md](PLAN_guardrails.md) — Guardrails implementation plan
- [PLAN_crash.md](PLAN_crash.md) — WSL crash analysis
- [PLAN_diagnostics.md](PLAN_diagnostics.md) — Diagnostic logging plan
- [PLAN_verify_index.md](PLAN_verify_index.md) — Index verification plan