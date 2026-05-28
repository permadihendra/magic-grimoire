# Magic Grimoire — `/qna` Bug Fix & Gemini Tool Plan

> **Status:** Proposed

---

## Bug 1: `str object has no attribute 'partial_format'`

### Root Cause

`query_with_sources(mode="qna")` passes the result of `get_qna_prompt()` as `summary_template` to TreeSummarize. This is a plain Python string. LlamaIndex's TreeSummarize internally calls `.partial_format()` on the template with `query_str` and `context_str` variables — but plain strings don't have this method. Only `PromptTemplate` objects do.

**Why doesn't quiz/summary mode have this issue?**

They do — but LlamaIndex might handle them differently depending on the installed version. The error may be intermittent or version-specific.

### Fix

In `engine.py`, wrap the qna prompt in a `PromptTemplate` before passing to TreeSummarize:

```python
elif mode == "qna":
    from app.rag.prompts import get_qna_prompt
    from llama_index.core.prompts import PromptTemplate
    prompt_template = PromptTemplate(
        get_qna_prompt(count=count, existing_pairs=existing_pairs)
    )
```

This creates a proper `PromptTemplate` object that TreeSummarize can call `.partial_format()` on.

---

## Bug 2: `/qna` Not Recognized by Gemini

### Root Cause

The brain handler's `_TOOL_REGISTRY` has no `qna` entry. Gemini doesn't know this tool exists.

### Fix: Add the full tool chain (same pattern as `/quiz`)

**File 1: `app/plugins/study/handler.py`** — Add `generate_qna()` module-level function:

```python
async def generate_qna(topic: str, count: int = 10, chat_id: int | None = None) -> str:
    """Generate comprehension Q&A pairs on a topic.

    Used by BrainPlugin's qna() tool.
    """
    global _rag_engine, _doc_indexer

    if _rag_engine is None or _doc_indexer is None:
        return "⚠️ RAG engine not initialized. Restart the bot."

    try:
        index = await _doc_indexer.ensure_index()
        _rag_engine.set_index(index)

        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        row = await cursor.fetchone()
        if not row or row["cnt"] == 0:
            return "📭 No documents indexed yet! Run `/index` first."

        logger.info("generate_qna: %s (count=%d)", topic[:80], count)
        enhanced = f"Generate {count} Q&A pairs about: {topic}"

        # Retrieve passages first (for source metadata)
        passages = _rag_engine.retrieve_only(enhanced)
        chunk_count = len(passages)

        # Use engine's query_with_sources(mode="qna") — TreeSummarize flow
        result = await _rag_engine.query_with_sources(
            enhanced, mode="qna", count=count,
            chat_id=chat_id,
        )
        answer_text = result.get("answer", "") if isinstance(result, dict) else str(result)

        # Append source metadata
        if passages and answer_text:
            short_names = []
            seen = set()
            for p in passages[:3]:
                from app.rag.engine import _lookup_display_name
                sn = _lookup_display_name(p["filename"])
                if sn not in seen:
                    seen.add(sn)
                    short_names.append(sn)
            if short_names:
                src_line = "\n\n📖 *Sources used:* " + " · ".join(f"_{s}_" for s in short_names)
                answer_text += src_line

        answer_text += (
            "\n\n💡 *Next steps:* Want more questions? Send "
            f"`/qna {topic}` again for complementary pairs!"
        )

        chunk_count = len(passages)
        answer_text += (
            f"\n\n📊 *Retrieval:* {chunk_count} passages"
        )

        return answer_text
    except Exception as e:
        logger.error("generate_qna failed: %s", e, exc_info=True)
        return f"⚠️ Q&A generation failed: {e}"
```

**File 2: `app/plugins/brain/handler.py`** — Add `_tool_qna`:

```python
async def _tool_qna(topic: str, count: str = "10", chat_id: int | None = None) -> str:
    """Execute the qna() tool — generate comprehension Q&A pairs."""
    from app.plugins.study.handler import generate_qna
    try:
        n = max(3, min(20, int(count)))
    except (ValueError, TypeError):
        n = 10
    try:
        return await generate_qna(topic, n, chat_id=chat_id)
    except Exception as e:
        logger.error("qna() failed: %s", e)
        return f"⚠️ Sorry, Q&A generation failed: {e}"
```

**File 3: `app/plugins/brain/handler.py`** — Register in `_TOOL_REGISTRY`:

```python
_TOOL_REGISTRY = {
    "ask": _tool_ask,
    "quiz": _tool_quiz,
    "summarize": _tool_summarize,
    "qna": _tool_qna,           # NEW
    ...
}
```

**File 4: `app/plugins/brain/handler.py`** — Add to `_SLOW_TOOLS`:

```python
_SLOW_TOOLS = {"ask", "quiz", "summarize", "qna"}  # qna is slow
```

**File 5: `app/plugins/brain/handler.py`** — Add to `AGENT_PROMPT`:

Add a `qna()` tool description:
```
- TOOL: qna(topic, count="10") — Generate comprehension Q&A pairs.
  The tool output IS the Q&A pairs.
  Do NOT write your own Q&A before this tool.
  Use for: "create Q&A for exam prep", "test my understanding",
  "give me practice questions with answers".
```

**File 6: `app/rag/engine.py`** — Fix partial_format:

Wrap QNA prompt in PromptTemplate object.

---

## Summary

| Issue | Fix | Effort |
|---|---|---|
| `partial_format` error | Wrap qna prompt in `PromptTemplate` | 2 lines in engine.py |
| Gemini can't call `/qna` | Add `_tool_qna` + `generate_qna` + registry + prompt | ~60 lines across 2 files |

Ready to execute?
