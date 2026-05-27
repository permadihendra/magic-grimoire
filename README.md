# Magic Grimoire 📖✨ — RAG Study Agent

> Your personal Telegram study assistant powered by LlamaIndex + Ollama + Gemini.
> Ingest your PDFs and EPUBs, then query them conversationally — get answers, quizzes,
> summaries, and exam prep from YOUR documents.

**Stack:** Python 3.11+ · FastAPI · LlamaIndex · Ollama (magic-grimoire:3b) · nomic-embed-text · Gemini 3.1 Flash Lite · sqlite3

---

## Quick Start

```bash
# 1. Install Ollama + pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
ollama create magic-grimoire:3b -f Modelfile.3b

# 2. Clone + install
git clone <repo> magic-grimoire
cd magic-grimoire
cp .env.example .env
# Edit .env: set TELEGRAM_TOKEN and GEMINI_API_KEY
uv sync

# 3. Drop your study PDFs/EPUBs into docs/
cp my-book.pdf app/docs/

# 4. Launch!
./app/start-bot.sh
```

## Commands

| Command | Description |
|---|---|
| `/ask <question>` | Query your documents with AI |
| `/quiz <topic>` | Generate practice questions |
| `/index` | Index all documents in `app/docs/` |
| `/files` | List indexed files with Gemini-cleaned names |
| `/delete <id>` | Delete a file by its ID |
| `/status` | Show index statistics |
| `/help` | Show help |
| `/ping` | Health check |

**Free text:** Just type a question — Gemini decides the best tool.
No need to remember slash commands for most queries.

## Features

### 📚 Document Ingestion
- PDF (via LiteParse with OCR) and EPUB (via SimpleDirectoryReader)
- Auto-clean filenames via Gemini at index time
- Persistent vector index (LlamaIndex) for fast retrieval

### 🧠 Agentic Brain
- Gemini routes every message — intent detection in 50ms
- Automatic tool selection: ask, quiz, summarize, list_docs, chat
- Feedback learning: say "wrong source" and it penalizes that document
- Knowledge cache: 0.1s answers for repeated questions

### 🛡️ Stability on 8 GB GPU
- `magic-grimoire:3b` optimized model (1.9 GB weights, 4096 ctx, temp 0.0)
- OllamaGuard prevents concurrent operations (no VRAM crashes)
- Fresh sqlite3 per operation — no corruption from GPU load
- Auto-recovery from crashes

### 📊 Progress Visibility
- Phase-aware watcher reports at t=0.5s, 30s, 60s+
- "Found 3 passages from World Economy. Generating answer..."
- Escalating messages if query takes long
- "Primary source / Also" citation labels

## Architecture

```
Telegram → cloudflared → FastAPI → Gemini agent → Knowledge Cache (fast) → RAG Engine → answer
```

All inference runs locally via Ollama. Gemini handles the lightweight intent routing.

## Project Structure

```
magic-grimoire/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── database.py          # sqlite3 (fresh per op, no corruption)
│   ├── bot/                 # Telegram webhook + dispatcher
│   ├── plugins/
│   │   ├── brain/           # Gemini agent + tool router
│   │   ├── study/           # RAG backend
│   │   └── system/          # /start, /help, /ping
│   ├── rag/
│   │   ├── engine.py        # Query engine (group-sort, filtering)
│   │   ├── indexer.py       # Document ingestion (Gemini clean names)
│   │   ├── guard.py         # OllamaGuard (semaphore, health check)
│   │   ├── feedback.py      # FeedbackLearner (penalty/boost)
│   │   └── knowledge_cache.py # Q&A cache with cosine search
│   ├── ui/
│   │   └── progress.py      # ProgressWatcher (phase-aware)
│   └── llm/
│       └── gemini.py        # Gemini API client
├── data/                    # sqlite3 DB + LlamaIndex vector store
├── docs/                    # Your study documents go here
├── Modelfile.3b             # Optimized qwen2.5:3b config
├── start-bot.sh             # One-command launcher
└── pyproject.toml           # Python deps (uv-managed)
```

## Model

| Component | Model | VRAM |
|---|---|---|
| LLM | `magic-grimoire:3b` (optimized from qwen2.5:3b) | 1.9 GB |
| Embeddings | `nomic-embed-text` | 0.3 GB |
| Agent brain | Gemini 3.1 Flash Lite (cloud, free tier) | 0 |
| **Total** | | **~3.2 GB** (4.8 GB headroom on 8 GB card) |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Bot doesn't reply | Re-run `./app/start-bot.sh` (tunnel expired) |
| "database disk image is malformed" | Must! Delete `data/magic-grimoire.db` and restart |
| "Model not found" | `ollama create magic-grimoire:3b -f Modelfile.3b` |
| Index rebuild fails | Check Ollama is running: `ollama list` |
| Wrong source cited | Say "that's from the wrong book" — Gemini penalizes it |
