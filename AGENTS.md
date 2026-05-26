# Magic Grimoire — Agent Guide

## Project Overview

Telegram bot + RAG study agent powered by LlamaIndex and local Ollama models.
Query your documents conversationally — generate quizzes, summaries, and exam prep.

**Stack:** Python 3.11+ · FastAPI · LlamaIndex · Ollama · aiosqlite · python-telegram-bot

**Architecture:** Telegram webhook → FastAPI → Dispatcher → StudyPlugin → RAG Engine (LlamaIndex + Ollama)

---

## Repository Layout

```
magic-grimoire/
├── app/
│   ├── main.py              # FastAPI app factory + lifespan
│   ├── config.py            # pydantic-settings, Ollama-focused
│   ├── database.py          # aiosqlite — documents + query_history tables
│   ├── bot/
│   │   ├── gateway.py       # Telegram webhook receiver (FastAPI route)
│   │   ├── dispatcher.py    # Slash commands → plugins; free text → StudyPlugin
│   │   ├── middlewares.py   # Rate limiter + auth guard
│   │   ├── context.py       # BotContext dataclass builder
│   │   └── setup_webhook.py # One-time webhook registration script
│   ├── plugins/
│   │   ├── base.py          # Plugin ABC + PluginRegistry singleton
│   │   ├── system/          # /start, /help, /ping, /status
│   │   └── study/           # /ask, /quiz, /docs, /index + free text RAG
│   ├── rag/
│   │   ├── models.py        # Ollama LLM + embedding config
│   │   ├── indexer.py       # Document ingestion with LlamaIndex
│   │   ├── engine.py        # RAG query engine
│   │   └── prompts.py       # Study agent prompt templates
│   └── docs/                # Drop your PDFs/books here!
├── tests/                   # (placeholder)
├── pyproject.toml           # UV-managed, LlamaIndex + Ollama deps
└── .env.example              # Placeholder values only
```

---

## Plugin Architecture

Every plugin extends `Plugin` ABC from `app/plugins/base.py`:

```python
class Plugin(ABC):
    name: str
    commands: list[str]   # Telegram slash commands (without /)
    description: str

    async def handle(self, ctx: BotContext) -> str | None: ...
    async def on_load(self) -> None: ...
    async def on_unload(self) -> None: ...
```

**Rules:**
- Each plugin is a folder in `app/plugins/<name>/` with `handler.py`
- Use `logging` not `print()`

### StudyPlugin — RAG Query Agent 📖

The `study` plugin handles ALL study-related queries:

| Command/Input | Action |
|---|---|
| `/ask <question>` | Query indexed documents with RAG |
| `/quiz <topic>` | Generate practice questions |
| `/docs` | List indexed documents |
| `/index` | Re-index all documents |
| Free text | Auto-routed to RAG query |

**Key Features:**
- **Progress updates** — during long operations (indexing, querying), bot sends periodic edits to the "thinking" message showing what's happening
- **Model warm-up** — pre-loads LLM into VRAM on startup (8s warm-up vs 40s cold start)
- **Retry dedup** — ignores Telegram's duplicate message retries
- **Error feedback** — if something fails, user sees a helpful message with fix suggestions

**RAG Flow:**
```
User sends message
  ↓  "⏳ Searching the grimoire…" (thinking message)
StudyPlugin.handle()
  ↓  "📖 Loading study materials..." (progress update)
DocumentIndexer.ensure_index()  → loads/creates vector index
  ↓  "🔍 Searching documents..." (progress update)
RAGEngine.query(question, mode)
  ├─ "qa" → QA_PROMPT + context → Ollama response
  ├─ "quiz" → QUIZ_PROMPT + context → generated questions
  └─ "summary" → SUMMARY_PROMPT + context → summary
  ↓
Response → Telegram (edits thinking message)
```

### SystemPlugin ⚙️

| Command | Action |
|---|---|
| `/start` | Welcome message |
| `/help` | Show command list |
| `/ping` | Health check |
| `/status` | Bot + index statistics |

---

## RAG Architecture

### Models (`app/rag/models.py`)

```python
Ollama LLM:     qwen2.5:7b (default) — 4.7 GB, 8K context window
Ollama Embed:   nomic-embed-text    — 0.3 GB VRAM
```

**VRAM budget (8GB GPU):**
- Model weights: ~4.7 GB
- Embeddings: ~0.3 GB
- KV cache + context: ~1.5 GB (8K window)
- **Total: ~6.5 GB** — comfortable, 1.5 GB headroom

Configured via LlamaIndex's global `Settings` object.

### Warm-up (`app/rag/models.py`)

On startup, the model is pre-loaded into VRAM with a tiny 2-token chat request.
This avoids the 30-60s cold-start delay on the first real query.
Uses Ollama's AsyncClient directly with `num_predict=2`.

### Indexer (`app/rag/indexer.py`)

- Scans `app/docs/` directory for documents
- Uses `SimpleDirectoryReader` (supports PDF, txt, md, docx, epub)
- Creates `VectorStoreIndex` with Ollama embeddings
- Persists to `data/index_storage/` for fast reload
- Tracks documents in SQLite `documents` table

### Engine (`app/rag/engine.py`)

- Wraps `RetrieverQueryEngine` with custom study prompts
- Configurable `similarity_top_k` (default: 5)
- Supports query modes: `qa`, `quiz`, `summary`
- Returns both answer text and source citations

### Prompts (`app/rag/prompts.py`)

Three prompt templates:
- `QA_PROMPT` — General Q&A with document context
- `QUIZ_PROMPT` — Generate mixed-format practice questions
- `SUMMARY_PROMPT` — Concise topic summaries

All prompts inject `AI_PERSONALITY` from config.

---

## Database Schema

```sql
documents:
  id, filename, filepath (unique), file_size, chunk_count, indexed_at

query_history:
  id, chat_id, query, response, created_at
```

---

## Secret Safety Rules

Coding agent MUST enforce these:

| Rule | Detail |
|---|---|
| `.gitignore` first | Generated before any other file; verified before any git command |
| `shell=True` forbidden | Always `create_subprocess_exec` with explicit arg list |
| No threading | `asyncio` only; never `import threading` |
| Parameterized SQL | All DB writes use `?` placeholders; no f-string SQL |
| Secrets via `settings` | Every credential from `settings.*`; never hardcoded, never in comments |
| Secrets never logged | No `logging.debug(settings.*)` or equivalent |
| Explicit HTTP timeouts | Every `httpx` call sets `timeout=` |
| Log with `logging` | No `print()` in production paths |
| `.env.example` fake values | e.g. `your-telegram-bot-token-here`, never a real key format |

---

## Common Commands

```bash
uv sync                     # Install runtime deps
uv sync --extra dev         # Install dev deps
uv run uvicorn app.main:app --reload             # Dev server
uv run ruff check .         # Lint
uv run ruff check --fix .   # Auto-fix
uv run mypy app/            # Type check
uv run python -m app.bot.setup_webhook           # Register webhook
uv run python -m app.bot.setup_webhook --delete  # Remove webhook
```

---

## Ollama Setup

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended models
ollama pull qwen3:4b        # LLM (~3 GB)
ollama pull nomic-embed-text  # Embeddings (~0.5 GB)

# Run Ollama (if not running as service)
ollama serve
```

---

## Deployment Checklist

```bash
# 1. Install deps
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo> magic-grimoire
cd magic-grimoire
cp .env.example .env && chmod 600 .env
uv sync --no-dev
mkdir -p data app/docs

# 2. Add your documents
# Copy PDFs/text files to app/docs/

# 3. Start Ollama (ensure it's running)
ollama list  # verify models are pulled

# 4. Start the bot
uv run uvicorn app.main:app --host 127.0.0.1 --port 8123

# 5. Register webhook (in another terminal)
uv run python -m app.bot.setup_webhook

# 6. Index documents via Telegram
# Send /index to the bot

# 7. Start studying!
# /ask "What are the main topics in chapter 1?"
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Bot doesn't reply | Webhook not set | Run `uv run python -m app.bot.setup_webhook` |
| "Index rebuild failed" | Ollama not running | `ollama serve` |
| "Model not found" | Model not pulled | `ollama pull qwen3:4b` |
| Empty query results | No docs indexed | Check `app/docs/` → `/index` |
| Markdown formatting broken | Telegram Markdown limitations | Bot falls back to plain text |

---

## Related

- [PLAN.md](PLAN.md) — Development roadmap
- [LlamaIndex Docs](https://docs.llamaindex.ai/)
- [Ollama Models](https://ollama.com/library)
