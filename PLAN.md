# Magic Grimoire 📖✨ — RAG Study Agent

## Vision

A Telegram bot powered by LlamaIndex + local Ollama models that ingests your study documents (PDFs, books, notes) and lets you query them conversationally. Like having a personal study assistant that generates quizzes, flashcards, and exam prep from YOUR materials.

**Stack:** Python 3.11+ · FastAPI · LlamaIndex · Ollama (Qwen3:4b) · aiosqlite · python-telegram-bot

---

## Architecture

```
User: "Create 10 multiple choice questions about Python decorators"
  ↓
Telegram Webhook (Gateway)
  ↓
Dispatcher → no slash → StudyPlugin (replaces old BrainPlugin)
  ↓
RAG Engine (LlamaIndex)
  ├─ Retrieves relevant chunks from indexed docs
  └─ Ollama Qwen3:4b generates response
  ↓
Formatted QA → Telegram Reply
```

### Document Ingestion Flow

```
Your PDFs/Books
  ↓ Drop into app/docs/ (or via Telegram upload)
  ↓
Indexer (LlamaIndex)
  ├─ Chunk → Embed (nomic-embed-text via Ollama)
  └─ Store in vector index (local, persistent)
  ↓
Ready for Query
```

---

## What's Kept from ai-assistant-light

| Component | Status | Notes |
|---|---|---|
| FastAPI app + lifespan | ✅ Keep, simplify | Remove scheduler, simplify plugin loading |
| Telegram webhook (gateway) | ✅ Keep | Core communication layer |
| Dispatcher | ✅ Keep | Simplified — no pending state |
| Middlewares | ✅ Keep | Rate limiter + auth guard |
| BotContext | ✅ Keep | Simple data class |
| Plugin base + registry | ✅ Keep | Core pattern |
| Config (pydantic-settings) | ✅ Keep | Simplified — Ollama-focused |
| Database (aiosqlite) | ✅ Keep | Simplified — no migrations needed |
| SystemPlugin (/start, /help, /ping) | ✅ Keep | Updated content |
| .env.example | ✅ Keep | Simplified |

## What's Stripped

| Component | Removed | Reason |
|---|---|---|
| ALL BrainPlugin (handler, context) | 🗑️ | Replaced by StudyPlugin |
| WebSearchPlugin | 🗑️ | Not needed for study agent |
| ReminderPlugin | 🗑️ | Not needed |
| NotesPlugin | 🗑️ | Not needed |
| AgendaPlugin | 🗑️ | Not needed |
| SummarizerPlugin | 🗑️ | Not needed |
| ScriptRunnerPlugin | 🗑️ | Not needed |
| All LLM providers (anthropic, gemini, etc.) | 🗑️ | Using Ollama only |
| LLM router | 🗑️ | Direct Ollama calls |
| Scheduler / APScheduler | 🗑️ | No background reminders |
| Cost tracker | 🗑️ | Local model = no cost |
| Migration files (001-010) | 🗑️ | Fresh DB schema |
| PC power scripts | 🗑️ | Not relevant |

## What's New

| Component | Description |
|---|---|
| `app/rag/engine.py` | LlamaIndex query engine — retrieves + generates |
| `app/rag/indexer.py` | Document ingestion — chunk, embed, index |
| `app/rag/prompts.py` | Study agent prompt templates |
| `app/rag/models.py` | Ollama model configuration |
| `app/plugins/study/handler.py` | StudyPlugin — /ask, /quiz, /docs, /index |

---

## Commands

| Command | Description |
|---|---|
| `/ask <question>` | Query your documents |
| `/quiz <topic>` | Generate practice questions |
| `/docs` | List indexed documents |
| `/index` | Re-index all documents in `app/docs/` |
| `/status` | Show index stats (doc count, chunk count) |
| `/ping` | Health check |
| `/help` | Show help |

## Ollama Models (8GB GPU)

| Model | Size | Role | VRAM |
|---|---|---|---|
| **qwen2.5:7b** (Q4_K_M) ★ | 4.7 GB | LLM — answer generation | ⭐ Recommended |
| qwen2.5:3b (Q4_K_M) | 1.9 GB | LLM — fast fallback | ✅ Lightweight |
| **nomic-embed-text** (F16) | 0.3 GB | Embeddings (via Ollama) | ✅ Negligible |

**Total VRAM:** ~6.5 GB (with 8K context) — comfortable on 8 GB.

---

## Deployment

### `app/start-bot.sh` — Quick Tunnel Launcher

Located at `app/start-bot.sh` (not `scripts/` — that directory was removed).

**What it does:**
1. ✅ Verifies Ollama is running + required models are pulled
2. ✅ Starts cloudflared quick tunnel (background)
3. ✅ Parses tunnel URL from logs
4. ✅ Updates `.env` with new webhook URL
5. ✅ Registers webhook with Telegram
6. ✅ Verifies Ollama model responsiveness
7. ✅ Sends startup notification to your Telegram
8. ✅ Launches uvicorn (foreground)

**Usage:**
```bash
./app/start-bot.sh
```

**Note:** Sits in `app/` because that's the deploy directory for the bot components.

### Manual Deployment

For production with a permanent tunnel:
```bash
# Start Ollama as service
sudo systemctl enable --now ollama

# Run bot directly
uv run uvicorn app.main:app --host 127.0.0.1 --port 8123
```

---

## Implementation Status

### ✅ Phase 1: Foundation — COMPLETE
1. ✅ Project plan written
2. ✅ Repository cleaned — all unused files removed
3. ✅ pyproject.toml updated — LlamaIndex + Ollama deps
4. ✅ .env.example — Ollama-focused
5. ✅ config.py — Simplified pydantic-settings
6. ✅ database.py — Fresh schema (documents + query_history)
7. ✅ main.py — Simplified lifespan, StudyPlugin loaded
8. ✅ dispatcher.py — Routes to StudyPlugin
9. ✅ system handler — Updated help text
10. ✅ gateway.py — Simplified, no message persistence
11. ✅ start-bot.sh — Quick tunnel launcher with Ollama checks

### ✅ Phase 2: RAG Core — COMPLETE
12. ✅ app/rag/models.py — Ollama LLM + embedding config
13. ✅ app/rag/indexer.py — Document ingestion pipeline
14. ✅ app/rag/engine.py — RAG query engine (qa/quiz/summary)
15. ✅ app/rag/prompts.py — Study prompt templates
16. ✅ app/plugins/study/handler.py — StudyPlugin

### ✅ Phase 3: Integration & Deploy — COMPLETE
17. ✅ Imports verified
18. ✅ .env configured with Telegram token
19. ✅ Models: qwen2.5:7b + nomic-embed-text pulled
20. ✅ Bot started, webhook registered, working
21. ✅ First query tested — progress updates working

### ✅ Phase 4: Fixes & Polish — COMPLETE
22. ✅ Timeout increased to 300s (prevents premature timeout)
23. ✅ Model warm-up on startup (pre-loads into VRAM)
24. ✅ Progress updates via Telegram (see what's happening)
25. ✅ Retry dedup (ignores Telegram duplicates)
26. ✅ Context window limited to 8192 (saves VRAM)
27. ✅ Error feedback (user sees *why* something failed)
28. ✅ Model switched to qwen2.5:7b (fits 8GB GPU)
