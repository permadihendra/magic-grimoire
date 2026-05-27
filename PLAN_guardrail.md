# Magic Grimoire — Guardrail Plan

## Crash Points Identified

I traced every code path that touches Ollama. Here are ALL crash scenarios:

### 1. Concurrent Ollama Calls 🔴 Critical

```
Telegram delivers 2 /ask messages within 0.5s (double-tap)

Request 1: retriever.retrieve() → Ollama embed → loads nomic-embed-text
Request 2: retriever.retrieve() → Ollama embed → tries to load AGAIN
                        ↓
              VRAM doubles → llama runner crashes
                        ↓
              ALL models disappear from Ollama
                        ↓
              Must re-pull 4.7 GB
```

This is what happened today.

### 2. LLM + Embedding Overlap 🔴

```
Request 1: LLM generating answer (qwen2.5:7b in VRAM, 4.7 GB)
Request 2: Index rebuild → embedding (nomic-embed-text, 0.3 GB)
                        ↓
              8 GB GPU: 4.7 + 0.3 + KV cache (~1.5) = 6.5 GB ← OK
              BUT: concurrent embedding calls each create their own context
                        ↓
              Oversubscribed → crash
```

### 3. Index Rebuild While Querying 🔴

```
/auth: User runs /index
/auth: Another user sends /ask
                        ↓
              indexer.from_documents() embeds 500+ chunks
              engine.retrieve() embeds 1 query
                        ↓
              Ollama processes both → VRAM spike → crash
```

### 4. Large Embedding Batch 🟡

```
User indexes 10 PDFs → 3000 chunks → VectorStoreIndex.from_documents()
                               ↓
              embed 3000 chunks in one shot → Ollama memory spike
```

### 5. Ollama Service Death 🔴

```
Any of the above crashes Ollama → ollama serve process dies
                               ↓
              Models lost from Ollama's store
                               ↓
              Bot tries to call Ollama → hang/timeout → bot crashes
```

### 6. LiteParse OOM 🟡

```
User uploads scanned PDF with 500 pages → LiteParse OCR
                               ↓
              Tesseract creates huge temp files → RAM spike
```

### 7. DB Contention 🟡

```
Gateway → dispatch (writes query_history)
BrainPlugin → ask (reads documents table)
Indexer → rebuild (writes documents table)
                ↓
        SQLite busy → 5s default timeout → stuck
```

### 8. Gemini Rate Limit 🟢

```
1500 free requests/day. If bot used heavily → quota exceeded
                              ↓
              Gemini returns 429 → bot returns generic error
```

---

## Guardrail Design

### Guardrail 1: Global Ollama Mutex 🔴 CRITICAL

```python
# ONE semaphore for ALL Ollama operations
_ollama_sem = asyncio.Semaphore(1)

# Track what operation is running (for user feedback)
_current_operation: str | None = None

async def execute_with_guard(coro, op_name: str, timeout: float = 120):
    """Single entry point for ALL Ollama calls."""
    
    if _ollama_sem.locked():
        raise OllamaBusyError(
            f"⏳ Ollama is busy with: {_current_operation}\n"
            "Please wait until it finishes, then try again."
        )
    
    try:
        _current_operation = op_name
        async with _ollama_sem:
            return await asyncio.wait_for(coro, timeout=timeout)
    finally:
        _current_operation = None
```

### Guardrail 2: Pre-Call Health Check

```python
_last_health: tuple[float, bool] = (0, True)

async def ollama_is_alive(timeout: float = 2.0) -> bool:
    """Check if Ollama is serving. Cached for 5 seconds."""
    global _last_health
    now = time.time()
    if now - _last_health[0] < 5:
        return _last_health[1]
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{settings.ollama_base_url}/api/tags")
            alive = r.status_code == 200
    except Exception:
        alive = False
    
    _last_health = (now, alive)
    return alive
```

### Guardrail 3: User Feedback

| Scenario | User Sees |
|---|---|
| Another query in progress | `⏳ Already processing "ask about world economy". Wait...` |
| Ollama dead | `⚠️ Study engine unavailable. It'll auto-restart. Try in 30s.` |
| Query timeout | `⚠️ Query timed out (120s). Try a simpler question.` |
| Embedding batch limited | `📚 Processing in batches of 50 to stay stable...` |

### Guardrail 4: Embedding Batch Limit

```python
MAX_CHUNKS_PER_BATCH = 100

# In indexer: split into batches
for i in range(0, len(documents), MAX_CHUNKS_PER_BATCH):
    batch = documents[i:i+MAX_CHUNKS_PER_BATCH]
    # ... index batch
```

### Guardrail 5: Crash Detection + Recovery

On startup, verify Ollama is healthy. If crash detected mid-operation:
1. Log the error
2. Notify admin chat: "⚠️ Ollama crashed. Restarting..."
3. Attempt restart: `subprocess.run(["ollama", "serve"], ...)`
4. Wait 5 seconds
5. Check if models are there
6. If models gone: "⚠️ Models lost. Run `ollama pull qwen2.5:7b`"

### Guardrail 6: DB WAL Mode

Enable WAL mode on SQLite to allow concurrent reads while writing:
```sql
PRAGMA journal_mode=WAL;
```

### Guardrail 7: Gemini Quota Tracking

Count Gemini API calls per day. If approaching 1500 limit:
- Switch to Ollama-only mode for simple queries
- Reserve Gemini for complex/ambiguous intent routing

---

## Implementation Plan

### Files

| File | What |
|---|---|
| `app/rag/guard.py` | **NEW** — All guardrail functions: mutex, health check, busy detection |
| `app/rag/models.py` | Wrap `get_llm()` and `get_embed_model()` with guardrail checks |
| `app/rag/engine.py` | Wrap `query_with_sources()` Ollama calls in guard |
| `app/rag/indexer.py` | Add batch limit, wrap embedding calls |
| `app/plugins/brain/handler.py` | Handle `OllamaBusyError`, show feedback to user |
| `app/bot/gateway.py` | Handle timeout errors gracefully |
| `app/database.py` | Enable WAL mode |

### Guardrail Architecture

```
               ┌──────────────────────┐
               │   Ollama Guard       │
               │   (app/rag/guard.py) │
               │                      │
               │  _ollama_sem (1)     │
               │  ollama_is_alive()   │
               │  execute_with_guard()│
               └──────┬───────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   engine.py     indexer.py    models.py
   (LLM gen)    (embed batch)  (warmup)
                      │
        ┌─────────────┼─────────────┐
        ▼                           ▼
   BrainPlugin                  StudyPlugin
   (handles errors,            (handles errors,
    shows feedback)             shows feedback)
```
