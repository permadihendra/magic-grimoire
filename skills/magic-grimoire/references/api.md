# Magic Grimoire API Reference

## Base URL

```
http://localhost:8123/api
```

## Authentication

None required for local access. API is not exposed to the internet.

## Endpoints

### GET /api/health

Health check. Returns service status.

**Response:**
```json
{
  "status": "ok",
  "service": "magic-grimoire",
  "model": "magic-grimoire:3b",
  "agent": "gateway"
}
```

---

### POST /api/tools/ask

Query study documents with RAG. Returns answer with source citations.

**Request:**
```json
{
  "query": "Apa itu loss aversion?",
  "difficulty": "normal",  // "simple" | "normal" | "advanced"
  "document": null          // optional: focus on specific document
}
```

**Response:**
```json
{
  "result": "📖 *Primary source:* Finance for Normal People\n\nLoss aversion adalah...",
  "status": "ok"
}
```

**Notes:**
- `difficulty="simple"` → simpler explanation, good for beginners
- `difficulty="advanced"` → detailed, technical explanation
- `document` parameter filters to specific document name

---

### POST /api/tools/quiz

Generate practice questions on a topic.

**Request:**
```json
{
  "topic": "Bab 3 Kejawen",
  "count": 5,               // 1-20
  "difficulty": "normal"    // "simple" | "normal" | "advanced"
}
```

**Response:**
```json
{
  "result": "📝 Quiz: Bab 3 Kejawen\n\n1. Apa yang dimaksud dengan...\n   Jawaban: ...\n\n2. ...",
  "status": "ok"
}
```

---

### POST /api/tools/summarize

Create a topic summary from documents.

**Request:**
```json
{
  "topic": "meditasi dalam Kejawen"
}
```

**Response:**
```json
{
  "result": "📊 Ringkasan: Meditasi dalam Kejawen\n\n...",
  "status": "ok"
}
```

---

### POST /api/tools/qna

Generate Q&A pairs for study.

**Request:**
```json
{
  "topic": "Bab 5",
  "count": 10               // 3-20
}
```

**Response:**
```json
{
  "result": "❓ Q&A: Bab 5\n\nQ: Apa itu...?\nA: ...\n\nQ: ...",
  "status": "ok"
}
```

---

### GET /api/tools/list_docs

List all indexed documents. No request body needed.

**Response:**
```json
{
  "result": "📚 *Your Library* (3 documents indexed)\n\n1. Finance for Normal People.pdf...",
  "status": "ok"
}
```

---

### POST /api/tools/retrieve

Retrieve raw passages without LLM generation. Fast.

**Request:**
```json
{
  "query": "meditasi",
  "document": null           // optional: filter by document
}
```

**Response:**
```json
{
  "result": "**1.** Kejawen.pdf (score: 0.892)\n   Meditasi adalah...\n\n**2.** ...",
  "status": "ok"
}
```

---

### POST /api/tools/feedback

Submit feedback about retrieval quality.

**Request:**
```json
{
  "type": "good",           // "good" | "wrong_source" | "reset"
  "detail": "Finance for Normal People"
}
```

**Response:**
```json
{
  "result": "Got it! I've noted that 'Finance for Normal People' was helpful.",
  "status": "ok"
}
```

## Error Responses

All endpoints return 500 on internal errors:
```json
{
  "detail": "Error message"
}
```

404 when no answer found:
```json
{
  "detail": "No answer found"
}
```
