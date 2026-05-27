# Magic Grimoire — Improvement Plan #4: Error-Proof Slow-Path

## Problem
If the tool execution in the slow-path fails (timeout, OOM, exception), the commitment message stays stuck forever. The user sees "📍 Found passages... ~10+ min" with no resolution.

## Root Cause
The tool execution block has no try-except:

```python
# Current (broken — no error handling)
for t_name, t_params in tool_tasks:
    result = await tool_fn(**t_params)  # ← can throw!
    result_lines.append(result)
```

If this throws:
- Exception propagates up → dispatch → gateway
- dispatch returns None → no message sent
- Commitment message hangs permanently

## Fix
Wrap tool execution in try-except. On failure, edit the commitment message with a clear error notification:

```python
for t_name, t_params in tool_tasks:
    tool_fn = _TOOL_REGISTRY.get(t_name)
    if tool_fn:
        try:
            result = await tool_fn(**t_params)
            result_lines.append(result)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", t_name, e, exc_info=True)
            # Edit commitment message with error
            if thinking_id:
                await _edit_message(chat_id, thinking_id,
                    "⚠️ *Sorry, the analysis failed.*\n\n"
                    f"Error: `{e}`\n\n"
                    "Try:\n"
                    "• A simpler or more specific question\n"
                    "• Check Ollama is running (`ollama serve`)\n"
                    "• Run `/index` to rebuild the index")
            return f"⚠️ Tool execution failed: {e}"
```

## Also: Gateway Safety Net

Add a timeout to the dispatch call in gateway.py. If dispatch takes > 5 minutes, send a fallback message:

```python
try:
    reply = await asyncio.wait_for(dispatch(update, None), timeout=300)
except asyncio.TimeoutError:
    reply = "⚠️ Processing timed out after 5 minutes. Try a simpler question."
```

## Files to Change

| File | Change |
|---|---|
| `app/plugins/brain/handler.py` | Wrap tool execution in try-except; on error, edit commitment msg + return error |
| `app/bot/gateway.py` | Add timeout to dispatch call |
