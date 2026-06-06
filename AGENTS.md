# Magic Grimoire — Agent Guide

> **Purpose:** This is the canonical reference for any coding agent working on this project.
> Read this file first. It tells you how everything works, what's implemented, and where to find things.

## Project Overview

**What it does:** Telegram bot + RAG study agent. Ingest PDFs/EPUBs → query them conversationally,
get answers with source citations, generate quizzes, Q&A pairs, summarize topics.

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
│   │   │                   #   Tools: ask, quiz, summarize, qna, list_docs,
│   │   │                   #          chat, feedback, retrieve, delete
│   │   │                   #   Intercepts all free-text messages
│   │   └── __init__.py
│   │
│   ├── study/handler.py      # StudyPlugin — RAG backend
│   │   │                   #   Slash: /ask, /quiz, /qna, /docs, /index, /files, /delete, /summarize
│   │   │                   #   Functions used by BrainPlugin tools:
│   │   │                   #   ask_query(), generate_quiz(), summarize_topic(),
│   │   │                   #   generate_qna(), retrieve_passages(), _tool_list_docs()
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
│   ├── guard.py             # OllamaGuard (semaphore + async VRAM check + OOM recovery)
│   │   │                   #   Key: ollama_check_vram(), OllamaGuard.__aenter__()
│   ├── feedback.py          # FeedbackLearner (per-chat document penalty/boost)
│   ├── knowledge_cache.py   # Q&A pair cache (REMOVED FROM ACTIVE USE — kept for reference)
│   ├── embeddings.py        # FallbackEmbedding (tries Ollama → auto-switches to CPU all-MiniLM-L6-v2)
│   └── prompts.py           # QA / Quiz / Summary / QnA prompt templates (difficulty variants)
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
        ┌─ ask()      → study/handler.ask_query() → engine.query_with_sources()
        ├─ quiz()     → study/handler.generate_quiz()
        ├─ summarize()→ study/handler.summarize_topic()
        ├─ qna()      → study/handler.generate_qna()
        ├─ list_docs()→ brain/handler._tool_list_docs_v2()
        ├─ feedback() → rag/feedback.py FeedbackLearner
        ├─ retrieve() → rag/engine.retrieve_only()
        └─ chat()     → Ollama direct (guarded by OllamaGuard)
              ↓
        Response edited into Telegram "thinking" message
```

---

## Key Files

### `app/plugins/brain/handler.py` — Agent Brain
- **`AGENT_PROMPT`** (system prompt at top): Tells Gemini how to route intents, tool syntax, rules
- **`_tool_ask()`** — calls `ask_query()` → `query_with_sources()` → returns `(answer, follow_up)` tuple
- **`_tool_quiz()`** — calls `generate_quiz()` with source info + next-step + stats footer
- **`_tool_summarize()`** — calls `summarize_topic()` with source info + next-step + stats footer
- **`_tool_qna()`** — calls `generate_qna()` with source info + next-step + stats footer
- **`_tool_list_docs_v2()`** — lists documents with word counts, parse method, probe status
- **`_tool_feedback()`** — records user feedback in FeedbackLearner per chat
- **`_tool_retrieve()`** — fast retrieval pass (no LLM), returns raw passages
- **`_tool_chat()`** — casual Ollama chat (**wrapped in OllamaGuard** — prevents concurrent calls)
- **`_conversation_memory`** — last 5 exchanges per chat for context
- **`_TOOL_REGISTRY`** dict — maps 9 tool names to handler functions
- **`_SLOW_TOOLS = {"ask", "quiz", "summarize", "qna"}`** — trigger thinking indicator
- If no `TOOL:` lines in Gemini response → appends "ℹ️ This answer came from my training data" footer
- **Follow-up ordering:** Brain handler sends answer chunks first, then follow-up messages last

### `app/plugins/study/handler.py` — RAG Backend
- **`ask_query()`** — main RAG query: ensure_index → query_with_sources → returns `{answer, follow_up}`
- **`generate_quiz()`** — retrieve_only (for sources) → query(mode="quiz") → append sources + next-step + stats
- **`summarize_topic()`** — same pattern as generate_quiz but mode="summary"
- **`generate_qna()`** — generate_qna(topic, count, chat_id) → Q&A pairs with source info
- **`_handle_index()`** — 3-phase: pre-check (test-parse each file) → build index → post-build probe + per-file report
- **`_handle_qna()`** — pre-retrieval preview → Q&A generation with generic examples
- **`_handle_docs()`** — delegates to `_tool_list_docs_v2()`
- **`_handle_files()`** — lists files on disk with indexed status + metadata from DB
- **`_handle_delete()`** — deletes file, shows impact (word/chunk count from DB before deleting)
- **`_doc_indexer`** — module-level `DocumentIndexer` singleton
- **`_rag_engine`** — module-level `RAGEngine` singleton

### `app/rag/engine.py` — Query Engine
- **`RAGEngine.query_with_sources()`** — the main method:
  1. Check index health (L1 guardrail)
  2. Sync retrieval + document boosting + feedback adjustment
  3. Post-retrieval filtering: force named-doc chunks into context
  4. L1b: 0 chunks → early return (no LLM call)
  5. Token budget check → early return if insufficient headroom
  6. L3-L6: OllamaGuard (180s) → TreeSummarize (120s timeout → fallback to simple prompt)
  7. L7: Validate answer length → generic "no answer" if too short
  8. Build source citations with score labels
  9. Add next-step suggestion 💡
  10. Add diagnostics footer 📊
  11. Return `{answer: str, sources: list[dict], follow_up: str}`
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
- **`get_llm()`** — Ollama LLM (magic-grimoire:3b), temperature=0.1, num_predict=2048, num_ctx=4096
- **`get_embed_model()`** — FallbackEmbedding(primary=OllamaEmbedding(embed_batch_size=3))
  - Tries Ollama nomic-embed-text first
  - On failure → auto-switches to CPU sentence-transformers/all-MiniLM-L6-v2
- **`warm_up()`** — sends 2-token request to pre-load LLM into VRAM on startup
- **`configure_settings()`** — sets LlamaIndex global Settings

### `app/rag/guard.py` — Ollama Guard
- **`ollama_check_vram()`** — async nvidia-smi call, returns (free_mb, used_mb, total_mb)
- **`OllamaGuard.__aenter__()`** — checks VRAM + acquires semaphore
  - Blocks if >1 concurrent Ollama operation
  - Returns guard object; raises if <1800MB free VRAM or Ollama dead
- **`OllamaBusyError`, `OllamaDeadError`** — raised instead of generic exceptions
- **`verify_ollama_on_startup()`** — called in main.py lifespan, logs warning but doesn't crash

### `app/rag/embeddings.py` — Fallback Embedding
- **`FallbackEmbedding`** — wraps OllamaEmbedding + sentence-transformers
  - `_async_get_text_embedding()` — tries primary, falls back on error
  - `_async_get_text_embedding_batch()` — tries primary, falls back on error
  - `_aget_query_embedding()`, `_get_query_embedding()` — for retrieval-time embedding
  - `_using_fallback` property — true once sentence-transformers activated

### `app/rag/feedback.py` — Feedback Learning
- **`FeedbackLearner`** — per-chat document scoring
  - `penalize_document()` — reduces score for wrongly-cited docs
  - `boost_document()` — increases score for correctly-cited docs
  - `get_blocked_docs()` — returns set of docs to deprioritize
  - `adjust_score()` — called in engine after retrieval, before LLM

### `app/rag/prompts.py` — Prompt Templates
- **`QA_NORMAL`** / `QA_SIMPLE` / `QA_ADVANCED` — question answering with difficulty variants
- **`QUIZ_NORMAL`** / `QUIZ_SIMPLE` / `QUIZ_ADVANCED` — quiz generation
- **`SUMMARY_PROMPT`** — topic summarization
- **`QNA_PROMPT`** — comprehension Q&A pair generation (generic examples, no content anchoring)
- All templates include "vary between sessions" instructions for answer diversity

### `app/ui/progress.py` — Progress Watcher
- **`ProgressState`** — tracks phase, chunks_embedded, files_total, files_parsed, passges_found
- **`ProgressWatcher`** — asyncio task that sends Telegram edits at t=0.5s, then escalating intervals

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
  verified,           -- 1 if probe_index() passed after last build
  indexed_at          -- last re-index timestamp
)

query_history (id, chat_id, query, response, created_at)

knowledge_pairs (     -- KEPT for reference but no longer active in engine.py
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
| `/qna <topic> [count]` | `StudyPlugin._handle_qna()` | Q&A pairs + sources + next-step + retrieval stats |
| `/summarize <topic>` | `StudyPlugin._handle_summarize()` | Summary + sources + next-step + retrieval stats |
| `/docs` | `StudyPlugin._handle_docs()` | Document library with word counts, parse methods, probe status |
| `/index` | `StudyPlugin._handle_index()` | Pre-check → parse → embed → probe → per-file report |
| `/files` | `StudyPlugin._handle_files()` | All files on disk with indexed status + metadata |
| `/delete <id>` | `StudyPlugin._handle_delete()` | Deletes file + shows impact (size/words/chunks) |
| Free text | `BrainPlugin.handle()` | Gemini routes to appropriate tool |

**Free text routing:** Gemini decides the tool, StudyPlugin is only invoked when `/ask` etc. are typed explicitly. For free text, BrainPlugin's tools call study handler functions directly.

---

## Response Format Pattern

All RAG responses follow this structure:
```
[Answer text...]

📖 Primary source: The Bhagavad Gita     ← if document was found
📎 Also: Chapter Notes.txt               ← additional sources

💡 Next steps: Would you like a follow-up question, a `quiz` on this topic,
   or a `summary` of the key points? Just ask!

📊 Retrieval: 3 chunks | ~420 tokens | k=3 | ctx=4096  ← diagnostics footer
```

Quiz/summary/qna add `📖 Sources used:` between answer and next-step.

---

## Modelfile Parameters (magic-grimoire:3b)

```
temperature=0.1          # Slight variation for repeated questions
num_ctx=4096             # Context window for long answers
num_gpu=99               # GPU layers (NOTE: should be 1 for Q4_0 — see PLAN.md)
num_predict=2048         # Max output tokens
repeat_penalty=1.1       # Slight discourage of repetition
num_thread=4             # CPU thread count
```

**VRAM budget (RTX 3050 8GB):**
- LLM weights: 1.9 GB
- KV cache (4096 ctx): ~2.4 GB
- Embeddings (nomic): 0.3 GB
- **Total: ~4.6 GB** (3.4 GB headroom)

CPU fallback (sentence-transformers) uses 0 VRAM — available when Ollama embed fails.

---

## Guardrails (7 Layers)

| Layer | Location | Guards Against |
|---|---|---|
| L1 | `engine.query_with_sources()` | 0 chunks → early return, no LLM call |
| L1b | `engine.query_with_sources()` | Token budget <50 → early return |
| L2 | `engine.query_with_sources()` | 0 docs in index → early return |
| L3 | `Modelfile.3b` | num_ctx=4096, num_predict=2048, temp=0.1 |
| L4 | `guard.py OllamaGuard` | VRAM pre-check (async) before every Ollama call |
| L5 | `engine.query_with_sources()` | TreeSummarize 120s timeout → fallback prompt |
| L6 | `models.py warm_up()` | 2-token request pre-loads model into VRAM |
| L7 | `embeddings.py FallbackEmbedding` | Ollama embed crash → auto-switch to CPU |

**Additional safety:** `_tool_chat()` is wrapped in OllamaGuard — all Ollama calls go through the semaphore.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Fresh sqlite3 per operation | No background-thread corruption during GPU load |
| FallbackEmbedding | Ollama Go runner crashes during batch embed → use CPU all-MiniLM-L6-v2 |
| embed_batch_size=3 | Was 10 — too many concurrent calls crashed Ollama Go runtime |
| os.path.basename for file_name metadata | Compare reliably between os.listdir() vs stored metadata |
| tree_summarize over manual acomplete | Hierarchical synthesis; 120s timeout + fallback protects |
| num_ctx=4096 | Enough headroom for long answers with 3B model |
| Temperature 0.1 | Slight variation for repeated questions (was 0.0 — identical answers) |
| Gemini for agent routing | Fast (~50ms), free tier, better intent classification than Ollama |
| Detection footer for no-TOOL | If Gemini doesn't call a tool → "ℹ️ from training data" so user knows |
| Knowledge cache removed | Stale answers from cache; fresh generation every time |
| All Ollama calls through guard | `_tool_chat` included — prevents concurrent VRAM exhaustion |

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
| Wrong source cited | Say "that's from the wrong book" → feedback penalizes it |
| "Ollama busy" message | Another query is running — wait for it to finish |

---

## Related

- [PLAN.md](PLAN.md) — Development roadmap (canonical — all PLAN_*.md consolidated here)
