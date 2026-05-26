# Magic Grimoire 📖✨ — RAG Study Agent

> Your personal Telegram study assistant powered by LlamaIndex + Ollama.
> Ingest your PDFs and books, then query them conversationally — generate quizzes, summaries, and exam prep on demand.

**Version:** v0.1.0 | **Stack:** Python 3.11+ · FastAPI · LlamaIndex · Ollama (qwen2.5:7b) · python-telegram-bot

---

## Features

| Command | Description |
|---|---|
| `/ask <question>` | Query your indexed documents |
| `/quiz <topic>` | Generate practice questions with answer key |
| `/docs` | List all indexed documents |
| `/index` | Re-index all documents in `app/docs/` |
| `/status` | Show index stats and bot status |
| `/help` | Show help |
| `/ping` | Health check |

**Free text:** Just send any question — the bot auto-queries your documents.

---

## How It Works

```
Your PDFs/Books → app/docs/
       ↓
  LlamaIndex ingests → chunks → embeds → vector index
       ↓
  You: "Create 10 multiple choice questions about chapter 3"
       ↓
  RAG Engine retrieves relevant chunks + Ollama generates response
       ↓
  Formatted QA → Telegram Reply
```

### Local Models (8GB GPU)

| Model | Size | Role | VRAM |
|---|---|---|---|
| **qwen2.5:7b** ★ | 4.7 GB | LLM — answer generation | ⭐ Recommended |
| qwen2.5:3b | 1.9 GB | LLM — fast fallback | ✅ Lightweight |
| **nomic-embed-text** | 0.3 GB | Embeddings — document chunking | ✅ Negligible |

All inference runs locally via **Ollama** — no cloud API costs, no data leaves your machine.

> ⚡ **8K context window** configured to fit comfortably on 8GB GPU (~6.5GB total VRAM usage).

---

## Quick Start

### 1. Prerequisites

```bash
# Install Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull qwen3:4b
ollama pull nomic-embed-text

# Verify
ollama list
```

### 2. Get a Telegram Bot Token

Open Telegram, search for [@BotFather](https://t.me/BotFather), send `/newbot`, and save the token.

> ⚠️ **Important:** Go to Bot Settings → Group Privacy → **Disable** privacy mode so the bot can see all messages.

### 3. Clone & Configure

```bash
git clone <your-repo> magic-grimoire
cd magic-grimoire

cp .env.example .env
chmod 600 .env
# Fill in:
#   TELEGRAM_TOKEN      — from BotFather
#   TELEGRAM_WEBHOOK_URL — your public URL (Cloudflare/ngrok)
#   ALLOWED_CHAT_IDS    — your Telegram user ID
```

### 4. Install & Run

```bash
uv sync --no-dev
mkdir -p data app/docs
# Drop your PDFs/books into app/docs/
uv run uvicorn app.main:app --host 127.0.0.1 --port 8123
```

### 5. Register Webhook

```bash
uv run python -m app.bot.setup_webhook
```

### 6. Index Your Documents

Send `/index` to the bot. Then try:
```
/ask What are the key concepts in this material?
/quiz Python fundamentals
```

---

## Adding Documents

Place any supported file type in `app/docs/`:

| Format | Supported by default |
|---|---|
| PDF | ✅ Yes |
| Plain text (.txt) | ✅ Yes |
| Markdown (.md) | ✅ Yes |
| Word (.docx) | ✅ Yes |
| EPUB | ✅ Yes |
| Images (OCR) | ⚠️ Requires pymupdf |

After adding new files, run `/index` to rebuild the index.

---

## Architecture

```
Telegram ──► Cloudflare/ngrok ──► FastAPI webhook
                                       │
                                  ┌────┴────┐
                                  │ gateway │  ← secret token verification
                                  │  .py    │  ← rate limiter
                                  └────┬────┘
                                       │
                                  ┌────┴────┐
                                  │dispatch │  ← slash commands → free text → Study
                                  └────┬────┘
                                       │
                                  ┌────┴────┐
                                  │  Study  │  ← /ask, /quiz, /docs, /index
                                  │ Plugin  │  ← free text → RAG query
                                  └────┬────┘
                                       │
                              ┌────────┴────────┐
                              │   RAG Engine    │
                              │  (LlamaIndex)   │
                              │                 │
                              │  ┌───────────┐  │
                              │  │ Ollama    │  │
                              │  │ Qwen3:4b  │  │
                              │  │ nomic     │  │
                              │  └───────────┘  │
                              └─────────────────┘
```

---

## Development

```bash
uv sync --extra dev          # install including dev deps
uv run uvicorn app.main:app --reload  # hot-reload dev server
uv run ruff check .          # lint
uv run ruff check --fix .    # auto-fix
uv run mypy app/             # type check
```

---

## Ollama Model Guide

For the best balance of quality and speed on 8GB GPU:

| Model | Size | VRAM | Quality | Speed | Verdict |
|---|---|---|---|---|---|
| **qwen2.5:7b** ★ | 4.7 GB | ~6.5 GB total | Good ✅ | Fast 🚀 | ⭐ **Recommended** |
| qwen2.5:3b | 1.9 GB | ~3.5 GB total | Decent | Very Fast | 🆗 Lightweight |

To switch models:
```env
# In .env
OLLAMA_LLM_MODEL=qwen2.5:3b
```
Then `ollama pull qwen2.5:3b` and restart the bot.

---

## Security

- `.env` is NEVER committed — in `.gitignore` + `chmod 600`
- Parameterized SQL — all queries use `?` placeholders
- No `shell=True` anywhere
- Secrets never logged
- Webhook secret token — constant-time comparison
- `ALLOWED_CHAT_IDS` — restrict bot access to authorized users only
- Local models — your data never leaves your machine

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Bot doesn't reply | Webhook not set | `uv run python -m app.bot.setup_webhook` |
| Bot ignores group messages | Privacy mode enabled | BotFather → Disable Group Privacy |
| "No documents indexed" | app/docs/ is empty | Drop PDFs there → `/index` |
| Index rebuild fails | Ollama not running | `ollama serve` |
| "Model not found" | Ollama model not pulled | `ollama pull qwen2.5:7b` |
| Query stuck at "Searching passages" | VRAM overload | Switch to `qwen2.5:3b` in `.env` |
| Slow response | Model thrashing VRAM | Use smaller model or reduce `context_window` |

---

## License

MIT
