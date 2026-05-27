# Magic Grimoire — Improvement Plan #2: Empty Response Fix

## Problem
When users ask relevant questions about their documents, the bot sometimes returns empty or useless responses. Three independent failure modes cause this.

## Failure Mode A: RAG Engine Returns Empty

### Root Cause
`engine.aquery()` returns a response with no source nodes or empty text. This happens when:
- Documents failed to parse (LiteParse not installed, fallback also failed)
- Index contains dummy empty documents (created when all files fail to parse)
- Chunks exist but have no meaningful text

### Fix
In `app/rag/engine.py`, `query_with_sources()`:

```python
# After engine.aquery():
if not response.source_nodes:
    return {
        "answer": "📭 *No relevant passages found.*\n\n"
                  "Try a different question or upload documents "
                  "that cover this topic.",
        "sources": [],
    }

answer = str(response)
if len(answer.strip()) < 20:
    answer = "📭 *I couldn't find a clear answer in your documents.*\n\n" \
             "This might mean:\n" \
             "• The documents don't cover this topic\n" \
             "• The documents failed to parse correctly\n" \
             "• Try `/index` to re-index or upload new material"
```

---

## Failure Mode B: Edit Message Fails Silently

### Root Cause
In BrainPlugin's slow-path, if `_edit_message()` fails (Telegram rate limit, markdown error, network), it returns `False`. The code ignores this and returns `None`. The thinking message "📖 Searching..." stays forever. User sees nothing.

### Fix
In `app/plugins/brain/handler.py`, slow-path:

```python
# Current (broken):
if thinking_id:
    await _edit_message(chat_id, thinking_id, final_reply)
return None  # If edit fails, user is stuck

# Fixed:
if thinking_id:
    ok = await _edit_message(chat_id, thinking_id, final_reply)
    if not ok:
        return final_reply  # Fallback: gateway sends as new message
return None  # Edit succeeded
```

---

## Failure Mode C: Tool Response Too Short

### Root Cause
Ollama generates a very short/unhelpful response like "I don't know" (5-10 chars). This passes through all validation and gets shown as the "answer."

### Fix
In `app/plugins/brain/handler.py`, `_tool_ask()`:

```python
result = await ask_query(query, difficulty=difficulty)
if len(result.strip()) < 20:
    result = (
        "📭 *I searched your documents but couldn't find a good answer.*\n\n"
        "Suggestions:\n"
        "• Try different keywords\n"
        "• Upload more relevant documents\n"
        "• Use `/files` to check your available docs\n"
        "• Run `/index` if you recently added files"
    )
return result
```

### Agent Prompt
Add to `AGENT_PROMPT` in `brain/handler.py`:

> *"If a tool returns 'not found' or an empty/short answer: acknowledge it, don't pretend you found something. Suggest: different keywords, upload relevant docs, or check /files."*

---

## Files to Change

| File | Changes |
|---|---|
| `app/rag/engine.py` | `query_with_sources()`: validate source_nodes and answer length |
| `app/plugins/brain/handler.py` | Slow-path: fallback on edit failure. `_tool_ask()`: check response length. Agent prompt: handle not-found. |
