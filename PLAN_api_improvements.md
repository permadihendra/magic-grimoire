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

### Fix 3: Improve Q&A Prompt for Comprehension

Current prompt asks for "DEEP understanding" but doesn't explicitly exclude:
- Historical/trivia questions
- Questions about the book's structure
- Questions about commentators

**Add to QNA_PROMPT:**
```
- DO NOT ask about: chapter numbering, verse counts, commentators, or book structure
- DO ask about: Krishna's teachings, their meaning, and how they apply to life
- Questions should test if someone UNDERSTOOD the teaching, not if they READ the book
```

---

## Priority Order

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Add `document` param to quiz/summarize/qna | Medium | High |
| P1 | Improve retrieval (filter glossary) | Medium | High |
| P2 | Improve Q&A prompt | Small | Medium |

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
