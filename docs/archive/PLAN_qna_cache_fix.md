# Magic Grimoire — QnA Caching Audit & Fix Plan

> **Status:** Proposed  
> Same QnA question returns identical results every time.

---

## Root Cause: Knowledge Cache Returns Same Answer

`engine.py:276-278` checks the knowledge cache before every query:

```python
cached = await search_cache(question, document, chat_id)
if cached and cached["similarity"] >= 0.92:
    return cached["full_answer"]  # ← NO LLM CALL
```

For QnA, the question string is identical every time → cache hit → same result.
For `/ask`, similar questions get matched by embedding → cached result, no fresh generation.

The user's logic: Telegram already stores the conversation history. If the user asks the same question twice, they can scroll up. No need for a cache that returns stale answers.

## Fix: Remove Knowledge Cache from All Queries

Remove the cache check **and** cache store from `query_with_sources()` entirely. The cache system (`knowledge_cache.py`) stays for potential future use but is no longer consulted.

---

## Fix: Remove Knowledge Cache from All Queries﻿

### Change 1: Remove cache check (engine.py:276-278)

```python
# REMOVE these 3 lines:
from app.rag.knowledge_cache import search_cache
cached = await search_cache(question, document, chat_id)
if cached and cached["similarity"] >= 0.92:
    return { "answer": cached["full_answer"], ... }
```

### Change 2: Remove cache store (engine.py:557-560)

```python
# REMOVE these ~8 lines:
if chat_id is not None and len(answer.strip()) >= 100:
    try:
        from app.rag.knowledge_cache import store_pair
        asyncio.create_task(store_pair(...))
    except Exception:
        pass
```

### Change 3: Remove `_qna_history` (study/handler.py)

Remove module-level dict, history check, and storage. Every call is fresh.

---

## Files Changed

| File | Change |
|---|---|
| `app/rag/engine.py:276-281` | Remove cache check block |
| `app/rag/engine.py:557-565` | Remove cache store block |
| `app/plugins/study/handler.py` | Remove `_qna_history` (history check, storage) |

---

## After Fix

```
/qna krishna teachings  → generates 10 fresh pairs (LLM runs)
/qna krishna teachings  → generates 10 DIFFERENT fresh pairs (LLM runs again)
```

Every call is independent. Fresh generation every time.
