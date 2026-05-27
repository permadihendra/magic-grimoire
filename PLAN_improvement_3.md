# Magic Grimoire — Improvement Plan #3: Long-Processing UX

## Problem
Complex queries across large documents take significant time. Users see "📖 Searching..." with zero feedback about:
- How many documents/passages were found
- How large the corpus is
- Expected wait time
- Whether processing is still ongoing

## Solution: Commitment Flow

```
User: "key takeaways from world economy..."

Bot: 📖 Searching documents for relevant passages...
  ↓ (instant)

Bot: 📍 Found passages in 3 documents
     📚 Corpus: ~3 docs · ~600K words
     ⏳ Analyzing and formulating your answer... ~5 min
     I'll notify you when it's ready! ✅
  ↓ (actual processing: 1-5 min)

Bot: ✅ [full answer with sources]
```

## Implementation

### Step 1: Document Stats

Add `get_document_stats()` to `app/database.py`:
- Total documents in index
- Total file size (bytes)
- Estimated word count (file_size × 0.35 / 5)

### Step 2: Smart Slow-Path

In `app/plugins/brain/handler.py`, modify the slow-path before executing tools:

```python
if t_name == "ask" or t_name == "summarize":
    stats = await get_document_stats()
    if stats and stats["words"] > 20_000:
        estimate = _estimate_time(stats["words"])
        await _edit_message(chat_id, thinking_id, 
            f"📍 Found passages in **{stats['docs']} documents**\n"
            f"📚 Corpus: ~{stats['words']/1000:.0f}K words\n"
            f"⏳ Analyzing and formulating... ~{estimate}\n\n"
            f"I'll update you when it's ready! ✅")
```

### Step 3: Time Estimation

```python
def _estimate_time(total_words: int) -> str:
    if total_words < 50_000:    return "1 min"
    if total_words < 200_000:   return "2-3 min"
    if total_words < 500_000:   return "3-5 min"
    if total_words < 1_000_000: return "5-10 min"
    return "10+ min"
```

### Agentic Alignment

The agent prompt gets one rule:
> *"The ask() tool automatically estimates processing time for large documents. Trust the estimate — it's based on actual corpus size."*

No hardcoded messaging from Python beyond the stats. The actual conversational wrapping comes from Gemini's natural text before the TOOL line.

### Files to Change

| File | Change |
|---|---|
| `app/database.py` | Add `get_document_stats()` |
| `app/plugins/brain/handler.py` | Slow-path: show stats + estimate before execution, add `_estimate_time()` |
| `app/plugins/system/handler.py` | Update `/status` to use same stats for consistency |
