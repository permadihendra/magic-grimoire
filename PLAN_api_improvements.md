# OpenClaw API Integration — Improvement Plan

## Status: Phase 1 Complete ✅ → Phase 2 Improvements

---

## Issues Found

### 1. Document Filtering Missing on Most Endpoints

| Endpoint | Has `document` param? | Should? |
|----------|----------------------|---------|
| `ask` | ✅ Yes | ✅ |
| `retrieve` | ✅ Yes | ✅ |
| `quiz` | ❌ No | ✅ Yes |
| `summarize` | ❌ No | ✅ Yes |
| `qna` | ❌ No | ✅ Yes |
| `list_docs` | N/A | N/A |
| `feedback` | N/A | N/A |

**Impact:** User says "quiz from Bhagavad Gita" → quiz searches ALL documents, not just the Gita.

### 2. RAG Retrieval Prefers Glossary Over Content

When query uses topic keywords ("karma yoga dharma"), RAG pulls:
- Glossary definitions (low value)
- Commentary passages (academic)

Instead of:
- Krishna's actual dialogues (high value)
- Core teaching passages

**Root cause:** Keyword matching favors short definition chunks over narrative passages.

### 3. Q&A Quality Varies

- ✅ Good when retrieval gets actual content
- ❌ Weak when retrieval gets definitions/metadata

---

## Improvement Plan

### Fix 1: Add `document` Parameter to Quiz, Summarize, QnA

**Files to change:**
- `app/plugins/brain/handler.py` — add `document` param to `_tool_quiz`, `_tool_summarize`, `_tool_qna`
- `app/plugins/study/handler.py` — add `document` param to `generate_quiz`, `summarize_topic`, `generate_qna`
- `app/rag/engine.py` — add document filtering to query method
- `app/api.py` — add `document` field to `QuizRequest`, `SummarizeRequest`, `QnaRequest`

**Approach:**
```python
# In engine.py, add document filter to query
async def query(self, question: str, document: str = None):
    # If document specified, filter nodes to that document
    if document:
        nodes = [n for n in nodes if document.lower() in n.metadata.get("file_name", "").lower()]
```

### Fix 2: Improve Retrieval for Q&A Generation

**Approach A:** Boost narrative passages over glossary
- Add metadata flag for chunk type (narrative, definition, commentary)
- Boost score for narrative chunks when generating Q&A

**Approach B:** Use query refinement to expand topic queries
- Already have `query_refiner.py` — ensure it expands short topic queries into narrative-style queries
- Example: "karma yoga" → "Krishna teaches Arjuna about selfless action and performing duty without attachment"

**Approach C:** Two-pass retrieval for Q&A
1. First pass: retrieve top 20 candidates
2. Second pass: filter out glossary/definition chunks, re-rank remaining

### Fix 3: Adaptive Q&A Mode (Context-Aware)

**Don't force comprehension mode.** Let user intent decide.

**Approach: Detect mode from query keywords**

| User says | Mode | What it generates |
|-----------|------|-------------------|
| "quiz me on Bab 3" | trivia | Definitions, facts, recall questions |
| "quiz me deeply on Bab 3" | comprehension | Analysis, why, how, apply |
| "test my understanding" | comprehension | Deep questions |
| "soal tentang Bab 3" | trivia | Standard quiz |
| "buatkan quiz untuk ujian" | comprehension | Exam-style deep questions |

**Keyword triggers for comprehension mode:**
```
deeply, deep, understanding, understand, analyze, analyze,
comprehension, why, how does, explain, apply, apply to life,
untuk ujian, pemahaman, analisis
```

**Default mode: trivia** — definitions, facts, straightforward recall.

**Update QNA_PROMPT:**
```python
QNA_PROMPT_TRIVIA = """... Generate {count} trivia Q&A pairs. Focus on definitions, facts, key terms. Simple recall questions. ..."""

QNA_PROMPT_COMPREHENSION = """... Generate {count} comprehension Q&A pairs. Focus on WHY, HOW, ANALYSIS, APPLICATION. Deep understanding. ..."""
```

**Implementation in `app/rag/prompts.py`:**
```python
def detect_qna_mode(query: str) -> str:
    comprehension_keywords = [
        "deeply", "deep", "understanding", "understand",
        "analyze", "analisis", "comprehension", "pemahaman",
        "why", "how does", "explain", "apply", "untuk ujian"
    ]
    if any(kw in query.lower() for kw in comprehension_keywords):
        return "comprehension"
    return "trivia"
```

### Fix 4: Context-Aware Follow-ups

**Problem:** User says "quiz me" → gets quiz → says "make it harder" → system doesn't know which topic.

**Solution: Pass last query as context**

```python
# In app/api.py — add optional context fields
class QuizRequest(BaseModel):
    topic: str
    count: int = 5
    difficulty: str = "normal"
    document: str | None = None
    context: str | None = None  # Previous query for follow-ups
```

**When `context` is provided:**
- Use it to resolve ambiguous references
- "make it harder" + context="quiz on Bab 3" → quiz on Bab 3 with harder difficulty
- "from the other book" + context="quiz from Bhagavad Gita" → quiz from other book

**Implementation:**
```python
# In brain handler — resolve follow-up
def resolve_topic(topic: str, context: str | None) -> str:
    if not context:
        return topic
    # If topic is vague, use context
    vague = ["it", "that", "this", "harder", "easier", "more", "another"]
    if any(v in topic.lower() for v in vague):
        return context  # Fall back to previous topic
    return topic
```

**Example flow:**
```
User: "Quiz me on Bab 3"
  → topic="Bab 3", trivia mode
  → response includes: {"context": "Bab 3"}

User: "Make it harder"
  → topic="Make it harder", context="Bab 3"
  → resolved: topic="Bab 3", difficulty="advanced"
```

**API response always includes context for next call:**
```json
{"result": "...", "status": "ok", "context": "Bab 3"}
```

---

## Priority Order

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Add `document` param to quiz/summarize/qna | Medium | High |
| P1 | Improve retrieval (filter glossary) | Medium | High |
| P2 | Adaptive Q&A mode (trivia vs comprehension) | Small | Medium |
| P3 | Context-aware follow-ups | Medium | High |

---

## Testing

After fixes, verify:
```bash
# Quiz from specific book
curl -X POST http://localhost:8123/api/tools/quiz \
  -d '{"topic": "selfless action", "count": 3, "document": "Bhagavad Gita"}'

# QnA from specific book
curl -X POST http://localhost:8123/api/tools/qna \
  -d '{"topic": "Krishna teachings", "count": 5, "document": "Bhagavad Gita"}'

# Verify no glossary/definition questions in results
```
