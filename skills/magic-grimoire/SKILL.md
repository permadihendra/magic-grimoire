---
name: magic-grimoire
description: Study assistant tools via Magic Grimoire API. Use when user asks to query study documents, generate quizzes, summarize topics, create Q&A pairs, list indexed documents, or retrieve passages from knowledge base. Triggers on keywords like "quiz", "summarize", "study", "document", "rag", "grimoire", "belajar", "rangkuman", "soal", "ujian", "materi", "book", "books", "read", "notes". Also use when user asks about their books, notes, or study materials.
---

## Magic Grimoire API

REST API for study tools. Base URL: `http://localhost:8123/api`

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

## Query Refinement (Agent-Level)

**Before calling the API, refine vague queries for better retrieval.**

The RAG engine matches keywords to document chunks. Generic queries like "top themes" or "main topics" often fail because chunks contain specific content, not overviews.

### Rules

1. **Expand vague queries** into specific questions about content
2. **Add context words** that appear in the document (e.g., "customer experience" not just "CX")
3. **Use natural language** — "what are the main concepts discussed in" not just "concepts"
4. **Include document name** when user mentions a specific book

### Examples

| User says | Bad query | Good query |
|-----------|-----------|------------|
| "What's in this book?" | "topics in book" | "what are the main concepts and key themes discussed in this customer experience book" |
| "Summarize Chapter 3" | "chapter 3 summary" | "summarize the key teachings about meditation and dharma from Chapter 3 of the Bhagavad Gita" |
| "Quiz me on the book" | "quiz questions" | "generate practice questions about the main concepts from The Customer Experience Book" |
| "What does it say about karma?" | "karma" | "how does the Bhagavad Gita explain the concept of karma yoga and selfless action" |

### Document Name Matching

Use partial names — the API fuzzy-matches:
- `"Pennington"` → matches "Pennington_Alan_The_customer_experience_book..."
- `"Bhagavad Gita"` → matches "Bibek Debroy - The Bhagavad Gita-Penguin Books 2005.epub"
- `"Finance"` → matches "Statman_Meir_Finance_for_normal_people..."

List available documents first if unsure: `GET /api/tools/list_docs`

## Endpoints

### Query Documents (RAG)
```
POST /api/tools/ask
{"query": "...", "difficulty": "normal|simple|advanced", "document": "optional doc name"}
```

### Generate Quiz
```
POST /api/tools/quiz
{"topic": "...", "count": 5, "difficulty": "normal|simple|advanced", "document": "optional"}
```

### Summarize Topic
```
POST /api/tools/summarize
{"topic": "...", "document": "optional"}
```

### Generate Q&A Pairs
```
POST /api/tools/qna
{"topic": "...", "count": 10, "document": "optional"}
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

## Response Format

All tool endpoints return:
```json
{"result": "...", "status": "ok", "context": "topic for follow-ups"}
```

The `context` field can be passed back in follow-up queries to maintain conversation flow.

## Full Workflow

```
User asks about study documents
    │
    ▼
Agent: refine query (expand vague terms, add context)
    │
    ▼
Agent: check service health → start if needed
    │
    ▼
Agent: call API with refined query + document filter
    │
    ▼
Magic Grimoire: query refinement → retrieval → reranking → LLM generation
    │
    ▼
Agent: format and present answer to user
```

## Installation

Symlink from OpenClaw skills directory (auto-syncs with repo):

```bash
ln -sfn ~/my-projects/magic-grimoire/skills/magic-grimoire \
  ~/.openclaw/skills/magic-grimoire
```
