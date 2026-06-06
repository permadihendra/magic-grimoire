# Magic Grimoire — Follow-up Order Fix Plan

> **Status:** Proposed · **Priority:** Minor  
> Ensures follow-up (next-step + diagnostics) is always sent AFTER the main answer.

---

## Bug: Follow-up Sent Before Answer (Brain Handler Path)

### Current Order

```
Telegram chat for free-text "ask":

[Message 1] 📖 Searching documents...          ← thinking msg
[Message 2] 📊 Retrieval: 3 chunks...          ← FOLLOW_UP SENT TOO EARLY
[Message 3] Krishna teaches that the self...   ← main answer
```

`_tool_ask()` at brain/handler.py line ~237 sends follow_up via `_send_telegram_message` **before** returning the answer string. The brain handler then sends the answer chunks **after** the function returns.

### Correct Order

```
Telegram chat for free-text "ask":

[Message 1] 📖 Searching documents...          ← thinking msg (stays visible)
[Message 2] Krishna teaches that the self...   ← main answer
[Message 3] 📊 Retrieval: 3 chunks...          ← FOLLOW_UP SENT LAST
```

---

## Fix: Return Tuple, Send Follow-up After Chunks

### Change 1: `app/plugins/brain/handler.py` — `_tool_ask()` (line ~233)

```python
# OLD — sends follow_up early, then returns answer string
if follow_up and chat_id:
    await asyncio.sleep(0.5)
    await _send_telegram_message(chat_id, follow_up)
return answer

# NEW — return both, let brain handler send follow_up last
return answer, follow_up
```

### Change 2: `app/plugins/brain/handler.py` — Slow path tool loop (line ~538)

```python
# OLD — single result_lines list
result_lines = []
for t_name, t_params in tool_tasks:
    result = await tool_fn(**t_params)
    result_lines.append(result)

# After loop: send all chunks
chunks = _split_into_chunks(tool_text)
for chunk in chunks:
    await _send_telegram_message(chat_id, chunk)

# NEW — separate follow_ups, send after chunks
result_lines = []
result_follow_ups = []

for t_name, t_params in tool_tasks:
    result = await tool_fn(**t_params)
    if isinstance(result, tuple) and len(result) >= 2:
        result_lines.append(result[0])       # answer text
        result_follow_ups.append(result[1])  # follow_up text
    else:
        result_lines.append(result)
        result_follow_ups.append(None)

# After sending all answer chunks...
for fu in result_follow_ups:
    if fu:
        await asyncio.sleep(0.5)
        await _send_telegram_message(chat_id, fu)
```

### Change 3: `app/plugins/brain/handler.py` — Fast path (line ~590)

Same pattern for the fast path — if tool returns tuple, extract follow_up and send after.

### Not affected (already correct)

- **Slash command path** (`_handle_ask` → `DispatchResult.processing.follow_up` → gateway): ✅ Gateway already sends main chunks → follow_up last
- **Other tools** (`_tool_quiz`, `_tool_summarize`, `_tool_chat`): ✅ Return plain strings, no follow_up involved
- **`ask_query()` return type**: ✅ Still returns dict, `_tool_ask` unpacks it

---

## Bonus: User-Friendly Footer (Words, Not Just Tokens)

The diagnostics footer currently shows:
```
📊 Retrieval: 3 chunks | ~420 tokens | k=3 | ctx=4096
```

Keep all existing debugging info, just **add word counts alongside tokens** for non-technical understanding:
```
📊 Retrieval: 3 chunks | ~420 tokens (~315 words) | k=3 | ctx=4096
```

### Changes

**`app/rag/engine.py`** — Footer construction (line ~517-521):
```python
# OLD (tokens only)
diag = (
    f"📊 *Retrieval:* {len(chunk_texts)} chunks | "
    f"~{int(total_tokens)} tokens | k={settings.retrieval_top_k} | "
    f"ctx=2048"
)

# NEW (tokens + words)
# 1 token ≈ 0.75 words for English
diag = (
    f"📊 *Retrieval:* {len(chunk_texts)} chunks | "
    f"~{int(total_tokens)} tokens (~{int(total_tokens * 0.75)} words) | "
    f"k={settings.retrieval_top_k} | "
    f"ctx=2048"
)
```
---

## Files Changed

| File | Lines | Change |
|---|---|---|
| `app/plugins/brain/handler.py` | ~238 | Remove `_send_telegram_message` from `_tool_ask`, return `(answer, follow_up)` tuple |
| `app/plugins/brain/handler.py` | ~538-583 | Separate `follow_ups` list, send after main chunks |
| `app/plugins/brain/handler.py` | ~590-600 | Same for fast path |
| `app/rag/engine.py` | ~517-521 | Footer: tokens → words, add generated word count |
