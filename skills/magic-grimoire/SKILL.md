---
name: magic-grimoire
description: Study assistant tools via Magic Grimoire API. Use when user asks to query study documents, generate quizzes, summarize topics, create Q&A pairs, list indexed documents, or retrieve passages from knowledge base. Triggers on keywords like "quiz", "summarize", "study", "document", "rag", "grimoire", "belajar", "rangkuman", "soal", "ujian", "materi". Also use when user asks about their books, notes, or study materials.
---

## Magic Grimoire API

REST API for study tools. Base URL: `http://localhost:8123/api`

All endpoints accept JSON POST body unless noted. Responses: `{"result": "...", "status": "ok"}`

## Service Check

Before calling any tool, verify service is running:

```bash
curl -sf http://localhost:8123/api/health
```

If unreachable, start the service:

```bash
cd ~/my-projects/magic-grimoire && bash app/start-bot.sh &>/dev/null & sleep 3
curl -sf http://localhost:8123/api/health
```

If still unreachable after start, inform user: "Magic Grimoire is not running. Please check `cd ~/my-projects/magic-grimoire && bash app/start-bot.sh`"

### Health Check
```
GET /api/health
```

### Query Documents (RAG)
```
POST /api/tools/ask
{"query": "...", "difficulty": "normal|simple|advanced", "document": "optional doc name"}
```

### Generate Quiz
```
POST /api/tools/quiz
{"topic": "...", "count": 5, "difficulty": "normal|simple|advanced"}
```

### Summarize Topic
```
POST /api/tools/summarize
{"topic": "..."}
```

### Generate Q&A Pairs
```
POST /api/tools/qna
{"topic": "...", "count": 10}
```

### List Documents
```
GET /api/tools/list_docs
```

### Retrieve Passages (no LLM)
```
POST /api/tools/retrieve
{"query": "...", "document": "optional"}
```

### Send Feedback
```
POST /api/tools/feedback
{"type": "good|wrong_source|reset", "detail": "optional"}
```

## Usage

Use `curl` or `web_fetch` to call the API:

```bash
# Query documents
curl -X POST http://localhost:8123/api/tools/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Apa itu meditasi?", "difficulty": "normal"}'

# Generate quiz
curl -X POST http://localhost:8123/api/tools/quiz \
  -H "Content-Type: application/json" \
  -d '{"topic": "Bab 3", "count": 5}'

# List docs
curl http://localhost:8123/api/tools/list_docs
```

## Response Format

All tool endpoints return:
```json
{"result": "The actual content...", "status": "ok"}
```

If error:
```json
{"detail": "Error message"}
```

## Installation

Symlink from OpenClaw skills directory (auto-syncs with repo):

```bash
ln -sfn ~/my-projects/magic-grimoire/skills/magic-grimoire \
  ~/.openclaw/skills/magic-grimoire
```

Verify:
```bash
ls -la ~/.openclaw/skills/magic-grimoire
```
