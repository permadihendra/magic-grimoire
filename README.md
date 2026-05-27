# Magic Grimoire 📖✨ — RAG Study Agent

> Your personal Telegram study assistant powered by LlamaIndex + Ollama + Gemini.
> Ingest PDFs and EPUBs, then query them conversationally — get answers with source
> citations, generate practice quizzes, and summarize topics.

**Stack:** Python 3.11+ · FastAPI · LlamaIndex · Ollama (local) · sentence-transformers (CPU fallback) · Gemini 3.1 Flash Lite (agent brain) · sqlite3

---

## Quick Start

```bash
# 1. Ollama — pull + create model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama create magic-grimoire:3b -f Modelfile.3b

# 2. Clone + install
git clone <repo> magic-grimoire
cd magic-grimoire
cp .env.example .env        # Set TELEGRAM_TOKEN + GEMINI_API_KEY
uv sync                     # or: uv sync --extra epub --extra dev

# 3. Drop your PDFs/EPUBs into docs/
cp my-book.pdf app/docs/

# 4. Launch (always use this — manages cloudflared tunnel)
bash app/start-bot.sh
```

> ⚠️ **Always use `bash app/start-bot.sh`** — never `uvicorn` directly.

---

## Commands

| Command | What it does |
|---|---|
| `/ask <question>` | Query your documents — gets answer with source citations + next steps |
| `/quiz <topic>` | Generate practice questions with answers + sources |
| `/summarize <topic>` | Get a topic summary from your documents |
| `/docs` | See your library: file sizes, word counts, parse methods, probe status |
| `/index` | Pre-verify → parse → embed → probe: full index rebuild with per-file report |
| `/files` | All files on disk: indexed status, pending files, display names |
| `/delete <id>` | Delete a file — shows impact (size/words/chunks) before confirming |
| `/status` | Bot health + index statistics |
| `/help` | Command list |

**Free text:** Type any question — Gemini routes it automatically. No need to remember `/ask`.

---

## Features

### 📚 Document Ingestion
- **Multi-method parsing:** ebooklib → LiteParse (PDF+OCR) → Calibre CLI → SimpleDirectoryReader
- **Pre-index verify:** test-parses every file before building, shows word count or error
- **Probe verification:** runs a test retrieval after build to confirm index is usable
- **Per-file report:** each file shows parse method, word count, chunk count, probe status
- **Gemini filename cleaning:** converts "Bhagavad Gita-Penguin Books 2005.epub" → "The Bhagavad Gita"

### 🧠 Agentic Brain (Gemini)
- Routes every free-text message to the right tool in ~50ms
- Detects intent, selects tools, chains actions
- Conversation memory (last 5 exchanges per chat)
- Proactive suggestions: "Would you like a quiz or deeper explanation?"
- Detects when answer came from training data (no tool used) → shows footer

### 🔍 RAG Engine
- **Document-aware retrieval:** mentions a book name → boosts that book's chunks 2x
- **Feedback learning:** "wrong source" → penalizes that document for this session
- **Knowledge cache:** repeated questions answered in ~0.1s (no LLM call)
- **7-layer guardrails:** 0 chunks / token budget / Ollama busy → early returns, no crash
- **Fallback embedding:** Ollama embed crashes → auto-switches to CPU sentence-transformers

### 🛡️ Stable on 8 GB GPU (RTX 3050)

| Component | Model | VRAM |
|---|---|---|
| LLM | `magic-grimoire:3b` (qwen2.5:3b, 1024 ctx, temp 0.0) | 1.9 GB |
| Embeddings | `nomic-embed-text` (batch=3) | 0.3 GB |
| KV cache | 1024 context window | ~1.2 GB |
| **Total** | | **~3.4 GB** (4.6 GB headroom) |
| Fallback | `all-MiniLM-L6-v2` (CPU, sentence-transformers) | 0 GB |

**Guard layers:** VRAM pre-check → semaphore (1 concurrent op) → 30s synthesis timeout → fallback prompt → CPU embed fallback

### 📊 Progress Visibility
Every command shows structured output:
```
📚 Your Library (2 documents)

[1] 📄 **The Bhagavad Gita**
   └─ 📊 6.7 MB · 186 chunks · 46,677 words
   └─ ✅ ebooklib (EPUB) · probe verified · indexed 28 May 2026

[2] 📄 **World Economy Summary**
   └─ 📊 1.2 MB · 45 chunks · 8,340 words
   └─ ⚠️ SimpleDirectoryReader · probe unknown · indexed 27 May 2026

_Total: 2 documents · ~55,000 words across all materials_
```

---

## Architecture

```
Telegram → cloudflared → FastAPI → Gemini agent → tools
                                           │
        ┌──────────────┬───────────────────┴────────────────┐
        ↓              ↓                    ↓                   ↓
    ask_query()    generate_quiz()     list_docs()         chat()
        ↓              ↓                    ↓                   ↓
  engine.query_   engine.query       brain/handler       Ollama direct
  with_sources    (mode=quiz)         _tool_list_docs_v2
        ↓
  ┌─────┴─────┐
  ↓  L1-L7    ↓  TreeSummarize / fallback → answer + sources + stats
  guardrails
```

All inference runs locally via Ollama. Gemini handles lightweight intent routing only.

---

## Project Structure

```
app/
├── main.py              # FastAPI + lifespan (db init, plugin loading, warm-up)
├── config.py            # pydantic-settings (all secrets + tuning params)
├── database.py          # sqlite3 fresh per op — WAL mode, 5 tables
│
├── bot/                 # Telegram webhook + dispatcher + middleware
├── plugins/
│   ├── brain/handler.py # Gemini agent: tool router, system prompt, memory
│   ├── study/handler.py # RAG backend: ask_query, generate_quiz, index, docs
│   └── system/handler.py # /start, /help, /ping, /status
│
├── rag/
│   ├── models.py        # Ollama LLM + FallbackEmbedding config
│   ├── indexer.py       # Multi-method parser + vector index builder
│   ├── engine.py        # query_with_sources() with 7 guardrails
│   ├── guard.py         # OllamaGuard: VRAM check + semaphore + OOM recovery
│   ├── embeddings.py   # FallbackEmbedding: Ollama → CPU all-MiniLM-L6-v2
│   ├── feedback.py      # FeedbackLearner: per-chat document scoring
│   ├── knowledge_cache.py # Q&A cache with cosine search
│   └── prompts.py       # QA / Quiz / Summary prompt templates
│
├── ui/
│   ├── progress.py      # ProgressState + ProgressWatcher (phase-aware escalation)
│   └── helpers.py       # _build_file_card, _build_files_card, _fmt_timestamp
│
└── llm/
    └── gemini.py        # Gemini API client
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Bot doesn't reply | Re-run `bash app/start-bot.sh` (cloudflared tunnel expired) |
| "Model not found" | `ollama create magic-grimoire:3b -f Modelfile.3b` |
| Index rebuild fails | `ollama serve && ollama list` — check models are loaded |
| "Ollama embed crashed" | Bot auto-switches to CPU fallback — retry `/index` |
| "database disk image is malformed" | Delete `data/magic-grimoire.db` and restart |
| PC crashed during `/index` | CPU fallback (sentence-transformers) should prevent this |
| Wrong source cited | Say "that's from the wrong book" → feedback penalizes it |

---

## Development

```bash
uv run ruff check .             # Lint
uv run ruff check --fix .       # Auto-fix
uv sync --extra epub            # EPUB support
uv sync --extra dev             # Dev deps (pytest, mypy)

# Register/remove webhook
uv run python -m app.bot.setup_webhook
uv run python -m app.bot.setup_webhook --delete
```

For detailed code reference, see **[AGENTS.md](AGENTS.md)**.
For development plans, see **PLAN_*.md** files.