# Magic Grimoire — PLAN: Streaming + Pipeline Dashboard UX

## The Problem

Today's UX for heavy operations gives zero visibility into WHAT is happening and HOW LONG it will take:

```
Current /ask:
  📖 Searching documents...           (vague)
  🔍 Retrieved 3 passages            (step 1 done)
  🧠 Generating answer...            (starts, then...)
  ⏳ Still generating... (60s)       (timer fires — user gets nervous)
  ✅ [answer appears]                (finally — was it stuck?)
```

User has NO idea:
- What step they're on (1 of 3? 2 of 3?)
- How long is left
- Whether the system is still alive or crashed

## The Vision

### Streaming Output (the gold standard)

Every modern AI chat streams tokens. The user sees text appear live:

```
User: "what are key takeaways from world economy?"

Bot: 📖 Searching World Economy book...      (0.3s)
     🔍 Found 3 passages · 0.92 relevance    (0.5s)
     🧠 Generating answer...                 (starts streaming)

     The key takeaway from the World Economy book is the 
     concept of paradigm shift in the global financial 
     system. The author argues that traditional economic 
     models are breaking down...▌              (live streaming!)

     ✅ Generated 234 words in 12.4s
     📖 Source: World Economy (0.92)
```

### Pipeline Dashboard (for multi-step ops)

```
🔍 Processing query...

╭─ Step 1/3 ────────────────── ✅ Done (0.3s)
│  📖 Search World Economy
│  Found 3 passages
╰────────────────────────────────

╭─ Step 2/3 ────────────────── 🔄 Generating...
│  🧠 AI is writing the answer
│  ████████░░░░ 234/500 tokens (12s elapsed)
╰────────────────────────────────

╭─ Step 3/3 ────────────────── ⏳ Pending
│  📎 Adding citations + formatting
╰────────────────────────────────
```

### For /index

```
📚 Rebuilding index — 3 files

╭─ Phase 1/3 — Parse ────────── 🔄 Running...
│  ✅ Finance.pdf .............. 489 pages (2.1s)
│  ✅ World Economy.epub ....... 350 pages (1.8s)
│  🔄 Python Crash Course.pdf .. Parsing... (1.2s elapsed)
╰────────────────────────────────

╭─ Phase 2/3 — Embed ────────── ⏳ Pending
│  ~1,200 chunks to embed
│  Est: 30-60s based on corpus size
╰────────────────────────────────

╭─ Phase 3/3 — Save ─────────── ⏳ Pending
│  Persist to disk
╰────────────────────────────────
```

---

## Three-Tier Architecture

| Tier | What | Impact |
|---|---|---|
| **Tier 1: Streaming Output** | LLM tokens stream to Telegram live | 🔴 Game-changer — removes the "black box" feeling |
| **Tier 2: Pipeline Dashboard** | Multi-step ops show completed/running/pending | 🟠 Major — user knows where they are |
| **Tier 3: Adaptive Estimates** | Based on corpus size + past runs | 🟡 Polish — replaces useless timer with real estimate |

---

## Tier 1: Streaming Output

### How It Works

```python
# Current (blocking — user waits blindly)
response = await llm.acomplete(full_prompt)
answer = str(response)

# Streaming (user sees every token)
stream = await llm.astream_complete(full_prompt)

buffer = ""
last_update = time.time()
message_id = thinking_msg["message_id"]

async for token in stream:
    buffer += token.delta
    
    # Update Telegram every 0.5s or every 20 tokens
    if time.time() - last_update > 0.5 or len(buffer.split()) % 20 == 0:
        await edit_message(chat_id, message_id,
            f"🧠 Generating answer...\n\n{buffer}▌")
        last_update = time.time()

# Final update (without cursor)
await edit_message(chat_id, message_id, f"{buffer}\n\n{source_citations}")
```

### Technical Details

**Ollama streaming:** LlamaIndex's `Ollama` class wraps raw Ollama. It supports `astream_complete()` which returns an async iterator.

**Actually:** Let me verify. The `llama_index.llms.ollama.Ollama` class may or may not support streaming properly. Alternative: bypass LlamaIndex and use raw `ollama.AsyncClient`:

```python
import ollama
client = ollama.AsyncClient(host="http://localhost:11434")

stream = await client.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": full_prompt}],
    stream=True,
)

async for chunk in stream:
    token = chunk["message"]["content"]
    buffer += token
    # ... update Telegram
```

**Telegram rate limits:** 
- `editMessageText`: ~20/sec for bots
- But: long messages take longer to send
- Safe batching: edit every 500ms or every 15 tokens
- Add a 10-edit burst limit before falling back to 1s intervals

**Fallback:** If streaming fails for any reason → use non-streaming `acomplete()` (current behavior).

### Files Changed

| File | Change |
|---|---|
| `app/rag/engine.py` | Add `query_with_streaming()` method using `ollama.AsyncClient` |
| `app/plugins/brain/handler.py` | Slow-path: detect streaming support, use it for ask tool |

---

## Tier 2: Pipeline Dashboard

### ProgressTracker Class

```python
class ProgressTracker:
    """Track multi-step operation progress."""
    
    def __init__(self, chat_id: int, total_steps: int):
        self.chat_id = chat_id
        self.total_steps = total_steps
        self.steps: list[dict] = []
        self.message_id: int | None = None
    
    def add_step(self, name: str, description: str, status: str = "pending"):
        self.steps.append({
            "name": name,
            "description": description,
            "status": status,  # pending, running, done, error
            "started_at": None,
            "finished_at": None,
            "elapsed": None,
            "details": "",
        })
    
    async def start_step(self, step_index: int):
        step = self.steps[step_index]
        step["status"] = "running"
        step["started_at"] = time.time()
        await self._render()
    
    async def update_step(self, step_index: int, details: str):
        step = self.steps[step_index]
        step["details"] = details
        await self._render()
    
    async def finish_step(self, step_index: int):
        step = self.steps[step_index]
        step["status"] = "done"
        step["finished_at"] = time.time()
        step["elapsed"] = step["finished_at"] - step["started_at"]
        await self._render()
    
    async def _render(self):
        """Build dashboard Markdown and edit the Telegram message."""
        lines = []
        for i, step in enumerate(self.steps):
            icon = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌"}
            
            lines.append(f"╭─ Step {i+1}/{self.total_steps} — {step['name']} ── {icon[step['status']]}")
            lines.append(f"│  {step['description']}")
            
            if step["status"] == "done":
                lines.append(f"│  ✅ Completed in {step['elapsed']:.1f}s")
            elif step["status"] == "running" and step["details"]:
                lines.append(f"│  {step['details']}")
            
            lines.append("╰" + "─" * 30)
            lines.append("")
        
        text = "\n".join(lines)
        
        if self.message_id:
            await edit_message(self.chat_id, self.message_id, text)
        else:
            msg = await send_message(self.chat_id, text)
            self.message_id = msg["message_id"]
```

### Example Usage in Slow-Path

```python
tracker = ProgressTracker(chat_id, total_steps=3)

# Step 1: Retrieve
tracker.add_step("Search", "Searching documents for relevant passages")
await tracker.start_step(0)
passages = await retrieve(query)
await tracker.update_step(0, f"Found {len(passages)} passages")
await tracker.finish_step(0)

# Step 2: Generate
tracker.add_step("Generate", "AI is writing the answer")
await tracker.start_step(1)
answer = await generate_with_streaming(query, passages, tracker, step=1)
await tracker.finish_step(1)

# Step 3: Format
tracker.add_step("Format", "Adding citations and formatting")
await tracker.start_step(2)
formatted = format_answer(answer, sources)
await tracker.finish_step(2)

# Final: show answer
await edit_message(chat_id, tracker.message_id, formatted)
```

---

## Tier 3: Adaptive Time Estimates

### Learning from History

```python
# Track per-operation timing in a simple JSON file
{
    "ask_generation": {
        "samples": [8.2, 12.5, 9.1, 15.3],
        "avg_per_100_tokens": 3.2,
    },
    "index_embedding": {
        "samples_per_100_chunks": [25.1, 30.4],
        "avg_per_100_chunks": 27.8,
    }
}
```

### Estimate Formula

```python
def estimate_ask_time(corpus_words: int) -> str:
    """Estimate based on corpus size."""
    if corpus_words < 50_000:
        return "5-15s"
    elif corpus_words < 200_000:
        return "10-25s"
    elif corpus_words < 500_000:
        return "20-40s"
    else:
        return "30-60s"

def estimate_index_time(num_chunks: int) -> str:
    """Estimate based on past embedding times."""
    avg = _load_avg("index_embedding")
    if avg:
        seconds = avg * num_chunks / 100
        return f"{seconds:.0f}s"
    return f"~{num_chunks * 0.3:.0f}s"  # fallback: 0.3s/chunk
```

### Progressive Refinement During Streaming

```
🧠 Generating answer...
████████░░░░ 234/500 tokens
⏱ 12s elapsed · ~8s remaining
```

Token count comes from streaming. Total tokens estimated from prompt length.

---

## Implementation Order

| # | What | Files | Effort | Impact |
|---|---|---|---|---|
| 1 | **Streaming LLM output** | `engine.py`, `brain/handler.py` | ~2h | 🔴 Game-changer |
| 2 | **Pipeline Dashboard** | New: `app/ui/progress.py`, wire into slow-path | ~2h | 🟠 Major |
| 3 | **Streaming for /index** | `study/handler.py` + `indexer.py` callbacks | ~1.5h | 🟠 Major |
| 4 | **Adaptive estimates** | New: `app/ui/timing.py` | ~1h | 🟡 Polish |
| 5 | **Typing indicator** | `brain/handler.py` — send "typing..." action | ~15min | 🟢 Nice-to-have |

### Phase 1 (Streaming) — Technical Risk

**Risk:** LlamaIndex's `Ollama` class streaming support may be broken or unreliable.

**Mitigation:** Bypass LlamaIndex entirely for streaming. Use raw `ollama.AsyncClient`:

```python
import ollama

client = ollama.AsyncClient(host=settings.ollama_base_url)
stream = await client.chat(
    model=settings.ollama_llm_model,
    messages=[{"role": "user", "content": full_prompt}],
    stream=True,
    options={"temperature": 0.7, "num_ctx": 8192},
)

async for chunk in stream:
    token = chunk["message"]["content"]
    # ... append to buffer, update Telegram
```

This bypasses LlamaIndex's model wrapper entirely. The prompt template + context building stays in engine.py.

### Phase 2 (Dashboard) — Telegram Limitation

**Issue:** Telegram `editMessageText` has a practical limit on edit frequency. Too many edits = rate limited.

**Solution:**
- Max 1 edit per 500ms (configurable)
- Batch dashboard updates (only render when step status changes)
- Skip rendering if nothing changed
- If rate-limited: log warning, delay next update

### Phase 3 (Streaming /index) — Indexer Architecture

**Challenge:** `VectorStoreIndex.from_documents()` is LlamaIndex's internal call. We can't inject progress callbacks easily.

**Alternative:** The current code already batches with 50-doc limit. We can add inter-batch progress updates:

```python
for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i+BATCH_SIZE]
    
    # Update dashboard before batch
    await tracker.update_step(1, 
        f"Embedding batch {i//BATCH_SIZE + 1}/{num_batches} "
        f"({len(batch)} docs, est. {estimate} remaining)"
    )
    
    batch_index = VectorStoreIndex.from_documents(batch)
```

---

## Challenge: Can We Do Better?

### Approach A: WebSocket Streaming (Overkill for Telegram)

WebSocket would allow true real-time streaming without edit rate limits. But Telegram doesn't support WebSocket — we're limited to HTTP webhook + `sendMessage`/`editMessageText`.

### Approach B: Progressive Message Chain

Instead of editing one message, create a CHAIN of messages. Each step = new message. User sees the full history:

```
Message 1: 📖 Searching World Economy book...
Message 2: 🔍 Found 3 passages (0.3s)
Message 3: 🧠 The key takeaway is... (streaming)
Message 4: 📖 Source: World Economy (0.92)
```

**Pros:** No edit needed, natural thread, full history visible
**Cons:** More messages = more clutter

### Approach C: Hybrid — Dashboard + Streaming

The dashboard message stays (edited for step progress), and a SEPARATE message streams the answer:

```
Message 1 (edited): Dashboard showing step progress
Message 2 (new):      Streaming answer text
Message 3 (edited):   Final answer with sources
```

This is the BEST approach. Dashboard provides structure, streaming provides the "live" feel, final message replaces both.

### Approach D: Simple Timer with Real Progress

Even without streaming or dashboard, we can massively improve with:

```
📖 Searching (0.3s) ✅
🔍 Found 3 passages from World Economy
🧠 Generating... 0 words written, 45s elapsed
   → updates every 5s with word count from stream
🧠 Generating... 120 words written, 47s elapsed  
   → user sees it's making progress!
```

---

## Recommendation

**Phase 1 first** (Streaming — highest impact). Implement raw Ollama streaming bypass. User sees the answer being written live. This SINGLE change eliminates the "is it stuck?" anxiety.

**Phase 2 next** (Dashboard — structure). `ProgressTracker` class for multi-step ops. Beautiful step-by-step UI.

**Phase 3 last** (Estimates — polish). Learn from history, show refined estimates.
