# Magic Grimoire — QnA Caching Audit & Fix Plan

> **Status:** Proposed  
> Same QnA question returns identical results every time.

---

## Root Cause: Two Caching Layers

### Layer 1: Knowledge Cache (engine.py:276-278)

```python
cached = await search_cache(question, document, chat_id)
if cached and cached["similarity"] >= 0.92:
    return cached["full_answer"]  # ← NO LLM CALL, returns cached answer
```

For QnA, the question passed to the engine is:
```
"Generate 10 Q&A pairs about: krishna teachings"
```

This is the **exact same string** every time. Cache similarity is 1.0 (identical) → returns cached answer immediately. Model never runs.

**Affects:** `/qna krishna teachings` → cache hit → same result every time.

### Layer 2: `_qna_history` (study/handler.py:320-354)

Even if cache is bypassed, `_qna_history` stores previous pairs and includes them in the prompt as "do NOT repeat" context. The user explicitly said this is backfiring.

---

## Fix: Remove Both Caching Layers for QnA

### Change 1: Skip knowledge cache in QnA mode

```python
# engine.py:276 — add mode check
if chat_id is not None and mode != "qna":  # ← skip cache for QnA
    from app.rag.knowledge_cache import search_cache
    cached = await search_cache(question, document, chat_id)
    ...
```

### Change 2: Skip cache storage in QnA mode

```python
# engine.py:557 — add mode check
if chat_id is not None and len(answer.strip()) >= 100 and mode != "qna":
    asyncio.create_task(store_pair(...))
```

### Change 3: Remove `_qna_history` entirely

Delete the module-level dict, history check, and storage in `_handle_qna`. Every `/qna` call generates fresh independent results.

```python
# REMOVE these from study/handler.py:
# Line 36: _qna_history: dict[tuple[int, str], list[dict]] = {}
# Line 292: global ... _qna_history
# Line 320: existing = _qna_history.get(key, [])
# Line 354: _qna_history[key] = existing + new_pairs
```

### Change 4: Update `get_qna_prompt()` — remove existing_pairs param

Since `existing_pairs` is never used, simplify the prompt template.

---

## Files Changed

| File | Change |
|---|---|
| `app/rag/engine.py:276` | Add `mode != "qna"` to cache check |
| `app/rag/engine.py:557` | Add `mode != "qna"` to cache store |
| `app/plugins/study/handler.py` | Remove `_qna_history`, history check, history storage |

No change to `prompts.py` (existing_pairs param still accepted but never passed — harmless).

---

## After Fix

```
/qna krishna teachings  → generates 10 fresh pairs (LLM runs)
/qna krishna teachings  → generates 10 DIFFERENT fresh pairs (LLM runs again)
```

Every call is independent. Fresh generation every time.
