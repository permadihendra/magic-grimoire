# Magic Grimoire — Fix Plan: Split Query + Message Overflow + Silent Watcher

## Bug Chain Found

```
User: "key takeaways from world economy and financial paradigm books"

Gemini splits into TWO tool calls:
  TOOL: ask(query="...", document="World Economy")      → answer 1 (2000 chars)
  TOOL: ask(query="...", document="Financial Paradigm")  → answer 2 (2500 chars)

Combined final message = 4500 chars → Telegram 4096 limit → "message is too long" 400 error
→ User sees "Generating..." forever (edit fails, no fallback)
```

## Three Bugs

### Bug 1: Agent Splits Queries (Gemini)
Gemini treats "world economy and financial paradigm" as TWO documents to search 
separately. Should be ONE query — it's the same book.

**Fix:** Agent prompt rule + better document matching:
- Tell Gemini: "If the user mentions one book name, call ask() ONCE with document=bookname"
- Don't split "world economy and financial paradigm" — it's likely one book

### Bug 2: No Message Length Guard (Python)
No truncation before sending to Telegram. Combined response > 4096 chars = silent failure.

**Fix:** Truncate message to 4000 chars before editMessageText/sendMessage.
Add `... (truncated)` suffix.

### Bug 3: Watcher Reports Too Late (Python)
Watcher waits 60s before first report. If generation completes in 8-10s, user sees nothing.
Then response fails silently — user is stuck.

**Fix:** Watcher sends an immediate "I'm starting" report at t=0s.
Also: escalate faster — start reporting at 15s, not 60s.

---

## Fixes

### Fix 1: Agent Prompt — No Split Queries

```python
# Add to AGENT_PROMPT rules:
"CRITICAL: One ask() call per user question. Do NOT split a single question 
 into multiple ask() calls for different documents. If the user mentions 
 one book (even with a compound name like 'world economy and financial paradigm'), 
 call ask() ONCE with that as the document parameter."

# Improve document name matching:
"If the user says 'world economy and financial paradigm books':
 → document='World Economy' (just the main title, not word-for-word)"
```

### Fix 2: Message Truncation

```python
_MAX_TELEGRAM_LENGTH = 4000  # 4096 limit minus safety margin

def _safe_telegram_text(text: str, max_len: int = 4000) -> str:
    """Truncate text to fit Telegram message limit."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 20] + "\n\n... (truncated)"
```

Apply in:
- `_edit_message()` — before editing
- `_send_telegram_message()` — before sending
- Gateway fallback — before returning

### Fix 3: Watcher — Immediate Report + Faster Escalation

```python
async def _watch_loop(self):
    # Phase 1: Report immediately at t=0
    await asyncio.sleep(0.1)
    if not self.state.complete:
        msg = self._build_initial_report()
        await self._edit(msg)
    
    # Phase 2: Escalate faster — every 30s, then every 15s
    intervals = [30, 30, 15, 15, 15, 15]  # 30s, 60s, 75s, 90s, 105s, 120s
    for interval in intervals:
        await asyncio.sleep(interval)
        if self.state.complete:
            return
        msg = self._build_report()
        if msg:
            await self._edit(msg)
```

---

## Implementation

### Files to Change

| File | Change |
|---|---|
| `app/plugins/brain/handler.py` | Agent prompt: no-split rule. Add `_safe_truncate()`. Apply in slow-path before edit. |
| `app/ui/progress.py` | Watcher: immediate report at t=0 + faster escalation (30s/15s) |
| `app/bot/gateway.py` | Apply truncation to return values |
