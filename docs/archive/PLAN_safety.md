# Magic Grimoire — Safety Improvement Plan

> **Status:** Proposed · **Priority:** High  
> Discovered during concurrency audit (2026-05-28). Bot running, no changes made yet.

---

## Risk 1: `_tool_chat()` bypasses OllamaGuard ⚠️ HIGH

### Location
`app/plugins/brain/handler.py` lines 308-321

### Current code
```python
async def _tool_chat(text: str) -> str:
    from app.rag.models import get_llm
    try:
        llm = get_llm()
        response = await llm.acomplete(prompt)  # Direct Ollama call — no guard!
        return str(response).strip()
    except Exception as e:
        return "Hey! 😊 I'm here to help you study."
```

### Risk
If User A sends "what is karma?" (→ `ask()` → guarded) and User B sends "hello" (→ `chat()` → unguarded) simultaneously, both hit Ollama concurrently. Two concurrent Ollama LLM calls = VRAM exhaustion = crash.

### Fix
```python
async def _tool_chat(text: str) -> str:
    from app.rag.guard import OllamaGuard
    from app.rag.models import get_llm
    try:
        async with OllamaGuard("chat", require_health=False):
            llm = get_llm()
            response = await llm.acomplete(prompt)
            return str(response).strip()
    except OllamaBusyError:
        return "⏳ I'm busy answering another question. Try again in a moment!"
    except Exception as e:
        return "Hey! 😊 I'm here to help you study."
```

---

## Risk 2: `ollama_check_vram()` blocks event loop ⚠️ MEDIUM

### Location
`app/rag/guard.py` lines 117-133

### Current code
```python
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total",
     "--format=csv,noheader,nounits"],
    capture_output=True, text=True, timeout=5,
)
```

### Risk
`subprocess.run` is synchronous and blocks the event loop for up to 5 seconds. If `nvidia-smi` hangs (driver glitch), the entire bot hangs for 5 seconds — no webhook processing, no heartbeats, no progress updates.

### Fix
```python
proc = await asyncio.create_subprocess_exec(
    "nvidia-smi", "--query-gpu=memory.used,memory.total",
    "--format=csv,noheader,nounits",
    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
)
try:
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    result = stdout.decode().strip()
except asyncio.TimeoutError:
    proc.kill()
    return True, "VRAM check timed out"
```

---

## Risk 3: OllamaGuard semaphore TOCTOU race ⚪ LOW

### Location
`app/rag/guard.py` lines 137-144

### Current code
```python
if _ollama_sem.locked():          # Check
    raise OllamaBusyError(...)
await asyncio.wait_for(           # Use
    _ollama_sem.acquire(), timeout=5.0)
```

### Risk
Between `locked()` check and `acquire()`, another coroutine could slip through. This is a classic Time-of-Check-Time-of-Use race. However, `acquire()` is atomic — if the semaphore is taken, it blocks. The `locked()` check just provides a faster error path. No crash risk, just a slightly delayed "busy" message instead of instant.

### Fix (optional)
Remove the redundant `locked()` check and rely solely on `acquire()`:

```python
try:
    await asyncio.wait_for(
        _ollama_sem.acquire(), timeout=5.0)
except asyncio.TimeoutError:
    raise OllamaBusyError(
        f"Already processing: {_current_operation or 'unknown'}.")
```

---

## Summary

| Risk | Severity | Fix effort | Impact if not fixed |
|---|---|---|---|
| `_tool_chat()` no guard | 🔴 HIGH | 5 lines | Concurrent Ollama calls → VRAM crash |
| `subprocess.run` blocking | 🟡 MEDIUM | 10 lines | Bot hangs 5s on VRAM check |
| TOCTOU race | ⚪ LOW | 3 lines | Delayed "busy" message (milliseconds) |

## Priority for implementation

1. **HIGH** — `_tool_chat()` guard (direct crash risk)
2. **MEDIUM** — async nvidia-smi (bot freeze risk)
3. **LOW** — TOCTOU race (cosmetic, no crash)

Ready to execute?
