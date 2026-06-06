# Magic Grimoire — PLAN: WSL Crash Root Cause Analysis

## What We Know

WSL system crashed when running `"whats the bhagavad gita books about?"` via the brain plugin (Gemini agent → tool routing → RAG).

## Root Cause Analysis

### CRASH LOCATION: `_tool_ask` → `ask_query` → `_rag_engine.query_with_sources`

```
User message
  ↓
Gateway → BrainPlugin.handle()
  ↓
Gemini decides TOOL: ask(query="...")
  ↓
_tool_ask() → ask_query()
  ↓
_rag_engine.query_with_sources()
  ↓  ← CRASH HAPPENS HERE (or earlier)
Phase 1: sync retriever.retrieve()
Phase 2: async TreeSummarize llm.acomplete()
  ↓
Response back to Telegram
```

---

## Most Likely Causes (ranked by probability)

### 1. 🌶️🌶️🌶️ Ollama OOM — VRAM Heap Explosion (HIGHEST PROBABILITY)

**What:** `magic-grimoire:3b` runs out of VRAM while generating with TreeSummarize (which processes ALL retrieved chunks in a single synthesis call). With `num_ctx=2048`, if chunks are large or overlapping, the KV cache + attention computation exceeds the ~3.5-4GB budget → CUDA OOM → WSL freeze or crash.

**Why TreeSummarize is dangerous:**
- TreeSummarize does hierarchical summarization — it sends ALL chunks at once to the LLM
- With chunk_size=256, overlap=30, 3 chunks → ~800 tokens context
- BUT if the index has the same document indexed multiple times (or corrupted chunks), it could be way more
- Worst case: k=3, each chunk 256 tokens, overlap adds another 256 → 768 tokens just for context → plus system prompt + instructions → close to 2048 limit

**VRAM math:**
```
magic-grimoire:3b (Q4_0): ~1.9 GB in VRAM
KV cache (2048 ctx): ~1.2-1.5 GB
Embedding model (nomic): ~0.3 GB
─────────────────────────────────
Total: ~3.4-3.7 GB (on 8 GB GPU)
Headroom: ~4.3 GB

When TreeSummarize activates for synthesis → extra attention computation
If KV cache is large → OOM
```

**Evidence:** WSL system crash (not Python exception), which is characteristic of GPU OOM crashing the display driver or the entire WSL session.

**Fix:**
- Reduce `num_ctx` from 2048 → 1024 (halves KV cache)
- Add explicit `num_gpu=1` in Modelfile (not `num_gpu 99`) — 99 was meant for layered models, not quantized
- Keep `num_predict=256`
- Add explicit `num_thread=4` to prevent CPU oversubscription during GPU ops

### 2. 🌶️🌶️ CPU Oversubscription — WSL Host Overload

**What:** The bot runs on WSL2. During heavy LLM inference, CPU usage spikes. WSL2 shares CPU with the Windows host. If many cores are maxed, WSL may be killed by the Windows scheduler.

**Evidence:** The system didn't just crash the bot — WSL crashed entirely. This suggests a resource exhaustion at the WSL level, not a Python error.

**Why now (vs before):**
- We switched to `magic-grimoire:3b` — smaller but still runs inference
- We added TreeSummarize — which is CPU + GPU intensive
- The system has only 8GB RAM (some shared with Windows), 8 cores
- Multiple Python threads + Ollama + cloudflared + uvicorn → all competing for CPU

**Fix:**
- Pin Ollama to specific cores with `OLLAMA_THREADS=4`
- Reduce `num_thread` in Modelfile to 4 (matches 4GB VRAM allocation)
- Kill unnecessary processes before starting bot

### 3. 🌶️🌶️ LlamaIndex DocumentIndexer — Background Thread Conflict

**What:** The `ensure_index()` call in `ask_query()` might trigger a background indexing process if the index is stale or missing. This runs indexing (which is GPU-heavy) AND simultaneously tries to do retrieval → conflict.

**Evidence:** The index was rebuilt recently with new chunk settings (256/30). If the old index (512 chunk) is partially loaded, LlamaIndex might be re-chunking on-the-fly.

**Fix:**
- Add a check: if index is stale (>1 hour old or chunk_count doesn't match expected), reload it
- Add a flag `_index_loading` mutex so `ask_query` never triggers a re-index

### 4. 🌶️ TreeSummarize + Large Response → Telegram Timeout

**What:** If TreeSummarize generates a long response (~500+ tokens), the Telegram API timeout (300s gateway) may be hit, causing the bot to hang while Ollama keeps running.

**Why:** `num_predict=256` should limit this, but if the model goes off-prompt and generates extra content, it could still be long.

**Fix:**
- Cap the response: truncate at 2000 chars for Telegram compatibility
- Reduce `num_predict` further to 128

### 5. 🌶️ SQLite WAL Lock Contention

**What:** Multiple async operations hit the `_AsyncDB` at the same time. The `threading.Lock()` in `database.py` forces sequential access, but if the lock holder is slow (GPU busy), the next request waits... but `aiosqlite` is gone, so this shouldn't apply.

**Status:** Unlikely — we replaced `aiosqlite` with fresh `sqlite3` per operation. No background threads.

---

## Immediate Debug Plan (no execution until user confirms)

### Step 1: Add Diagnostic Logging — Find the Exact Crash Point

Before any fixes, add structured logs so we can pinpoint WHERE it crashes:

```python
# In _tool_ask() — BEFORE any heavy operation
logger.info(">>> _tool_ask START: query='%s' difficulty=%s document=%s",
            query[:50], difficulty, document)

# After ensure_index
logger.info(">>> Index loaded: %s chunks", index_size)

# Before llm call
logger.info(">>> TreeSummarize START: %d chunks, ~%d tokens",
            len(chunks), estimated_tokens)

# After llm call
logger.info(">>> TreeSummarize DONE: %d chars in %.1fs",
            len(answer), elapsed)
```

And wrap in try/except to catch the actual error:

```python
try:
    result = await ask_query(...)
    logger.info(">>> _tool_ask DONE")
except Exception as e:
    logger.error(">>> _tool_ask CRASHED: %s", e, exc_info=True)
    raise
```

### Step 2: Fix Modelfile — Prevent VRAM Explosion

```dockerfile
# In Modelfile.3b
PARAMETER num_ctx 1024      # HALVE: halves KV cache → more headroom
PARAMETER num_gpu 1         # Explicit: use 1 GPU layer for quantized model
PARAMETER num_thread 4      # Match CPU cores for WSL stability
PARAMETER num_predict 128   # HALVE: shorter responses, faster generation
PARAMETER temperature 0.0
PARAMETER stop "Answer:"
PARAMETER stop "<|im_end|>"
```

### Step 3: Reduce Ollama Base URL Health Check Frequency

The `ollama_is_alive()` health check runs before every Ollama call. On a slow WSL system, this might add latency. But more importantly — if the health check itself hangs (Ollama is stuck), we need a timeout.

Current: `httpx.AsyncClient(timeout=2.0)` — already has timeout. OK.

### Step 4: Kill TreeSummarize — Use Direct `acomplete` Instead

TreeSummarize is LlamaIndex's hierarchical summarizer. It's great for quality but expensive. For the crash investigation, replace it with a direct `acomplete()` call with a tighter prompt. This eliminates the synthesis overhead.

```python
# Instead of TreeSummarize
response = await llm.acomplete(full_prompt)  # where full_prompt is pre-built

# With:
full_prompt = f"""You are a study assistant.
Based ONLY on these excerpts, answer the question.

Excerpts:
{context}

Question: {question}

Answer:"""
```

### Step 5: Add explicit VRAM Monitoring

After the crash, add a post-crash check:

```python
import subprocess
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
    capture_output=True, text=True
)
logger.info("VRAM: %s", result.stdout)
```

This won't prevent the crash but helps us understand if VRAM was the bottleneck.

---

## Proposed Fixes Summary

| Priority | Fix | Why |
|---|---|---|
| **P1** | `num_ctx=1024` + `num_gpu=1` + `num_thread=4` in Modelfile | VRAM headroom |
| **P1** | Replace TreeSummarize with direct `acomplete` | Eliminate synthesis overhead |
| **P1** | Structured crash-point logging | Pinpoint exact failure location |
| **P2** | Reduce `num_predict` to 128 | Shorter responses = faster |
| **P2** | `ollama serve` with `OLLAMA_THREADS=4` | WSL CPU stability |
| **P3** | Index health check before retrieval | Prevent stale-index re-chunking |
| **P3** | Response truncation at 2000 chars | Telegram compatibility |

---

## Decision Needed Before Execution

**Question:** Do you want to:
- **(A) Diagnose first** — Add crash-point logging, test with simple query, see exactly where it fails, THEN fix
- **(B) Fix directly** — Apply all P1+P2 changes (num_ctx 1024, no TreeSummarize, simpler prompt) immediately, test, iterate

Option A is safer (we know exactly what's wrong).
Option B is faster (we address all likely causes at once).