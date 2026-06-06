# Magic Grimoire — QnA Prompt Fix & Interactivity Plan

> **Status:** Proposed  
> Two issues: 1) QnA always returns Krishna content regardless of topic  
> 2) QnA has no retrieval preview (unlike `/ask` flow)

---

## Bug 1: QnA Always Returns Krishna Content

### Root Cause

`QNA_PROMPT` in `prompts.py` has **hardcoded Krishna-specific examples**:

```python
Format:
[1] Q: What is the concept of dharma in the Bhagavad Gita?
    A: Dharma refers to righteous duty...
[2] Q: How does Krishna define karma yoga?
    A: Karma yoga is the path of selfless action...
```

The model treats these as content anchors, not just format examples. Even when the prompt says "based on the provided context" and the retrieved chunks are about Shakuni, the examples bias the model toward Krishna.

### Fix

Replace with **generic placeholder examples**:

```python
Format:
[1] Q: [Clear question about the topic from the documents]
    A: [Complete answer based only on the provided context]

[2] Q: [Another question on a different aspect of the topic]
    A: [Answer with details from the provided context]
```

This shows the format without anchoring on specific content.

---

## Improvement 2: Add Retrieval Preview to QnA

### Why

The brain handler's `/ask` flow shows:
1. "🔍 Searching documents..." → thinking msg
2. "🔍 Retrieved: Shakuni's role (0.85), ..." → progress update
3. "🧠 Generating answer..." → progress update
4. Final result → new messages

The `/qna` slash command just shows "📝 Generating Q&A pairs..." and waits silently.

### How

In `_handle_qna()`, add a pre-retrieval step to show passage names before generating:

```python
# Step 1: Show progress
await self._progress(ctx, "🔍 Searching documents for relevant passages...")

# Step 2: Pre-retrieve passages (fast) to show what's found
passages = _rag_engine.retrieve_only(topic, chat_id=ctx.chat_id)
if passages:
    names = seen_passages(passages[:3])
    await self._progress(ctx, f"🔍 Found passages about: {', '.join(names)}\n🧠 Generating Q&A pairs...")
else:
    await self._progress(ctx, "📝 Generating Q&A pairs...")

# Step 3: Full generation via engine
result = await _rag_engine.query_with_sources(topic, mode="qna", ...)
```

This shows the user what the bot found before it starts generating.

---

## Files Changed

| File | Change |
|---|---|
| `app/rag/prompts.py` | Replace Krishna examples with generic placeholders |
| `app/plugins/study/handler.py` | Add pre-retrieval preview in `_handle_qna()` |

Ready to execute?
