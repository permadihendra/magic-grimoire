# Magic Grimoire — Improvement Plan #5: Operation Log UX for All Commands

## Core Principle
Every long-running command follows the same pattern:
1. **Initial**: what's about to happen
2. **Steps**: real updates as each phase completes
3. **Timer**: periodic "still working" if any step takes > 60s
4. **Done**: final result

---

## Command Flows

### `/ask` and `/quiz` (via BrainPlugin)

```
📖 Searching documents...
  ↓ (retrieval: 0.1s)

🔍 Retrieved 3 passages:
   • Finance for Normal People (0.69)
   • World Economy (0.61)
  ↓

🧠 Generating answer...
  ↓ (if > 60s: timer fires every 60s)

⏳ Still generating... (60s elapsed)
  ↓ (if > 120s: another timer)

⏳ Still generating... (120s elapsed)
  ↓ (generation completes)

✅ [full answer with sources]
```

### `/index` (via StudyPlugin)

```
📚 Indexing 3 files...
  ↓ (parsing: ~2s/file)

📖 Parsing: Finance for Normal People.pdf
   ✅ 489 pages · 1.2M chars
📖 Parsing: World Economy.epub  
   ✅ 350 pages · 0.8M chars
📖 Parsing: Python Crash Course.pdf
   ✅ 200 pages · 0.4M chars
  ↓ (embedding: slowest phase)

🔄 Generating embeddings for ~1200 chunks...
  ↓ (if > 60s: timer fires)

⏳ Still embedding... (60s elapsed)
  ↓ (embedding completes)

💾 Saving index to disk...
  ↓

✅ Index rebuilt! 3 docs, ~1200 chunks in 35s
```

### Slash Commands (fast — no progress needed)
`/ping`, `/help`, `/status`, `/docs`, `/files`, `/delete` — all complete in < 1s. No progress needed.

---

## Implementation

### Shared Timer Function
Add to `app/plugins/brain/handler.py` (also used by StudyPlugin):

```python
async def _progress_timer(chat_id: int, message_id: int, 
                           label: str = "Processing",
                           interval: int = 60):
    """Send 'still working' updates every `interval` seconds."""
    import asyncio
    elapsed = 0
    try:
        while True:
            await asyncio.sleep(interval)
            elapsed += interval
            await _edit_message(chat_id, message_id,
                f"⏳ Still {label}... ({elapsed}s elapsed)")
    except asyncio.CancelledError:
        pass
```

### BrainPlugin Slow-Path Updates
- Split ask/quiz tool execution into: retrieve → show → generate with timer → show answer
- For `_tool_retrieve()`: show actual passages with scores

### StudyPlugin `/index` Updates
- Provide `_send_message()` and `_edit_message()` access in handler
- Phase 1: show file list
- Phase 2: parse files (show each as it completes)
- Phase 3: embedding with timer
- Phase 4: done

### Files to Change

| File | Change |
|---|---|
| `app/plugins/brain/handler.py` | Add `_progress_timer()`, update slow-path for ask/quiz, add `_tool_retrieve()` |
| `app/plugins/study/handler.py` | Update `_handle_index()` with phased progress + timer. Add `retrieve_passages()`. |
| `app/rag/engine.py` | Add `retrieve_only()` method for retrieval without LLM |
