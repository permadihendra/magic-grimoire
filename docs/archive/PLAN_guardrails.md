# Magic Grimoire — PLAN: Guardrails for Empty Index / Safe LLM Calls

## What We Know

1. **Empty index (0 chunks)** → `nodes = []` → `chunk_texts = []`
2. **TreeSummarize with empty chunks** → LlamaIndex may send FULL system prompt + query to Ollama (no context to constrain it)
3. **EPUB can't be parsed** → `ebooklib` not installed → index is always empty
4. **WSL crash** → likely Ollama OOM or CPU oversubscription during edge-case inference

## Guardrail Strategy: Defense in Depth

Multiple layers so if one fails, the next catches it.

---

## Layer 1: Pre-LLM Validation (engine.py)

Before calling TreeSummarize or any LLM:

```python
# ── Validate: empty or suspicious chunks ───────────────
chunk_texts = [n.text for n in nodes if n.text]
chunk_count = len(chunk_texts)

# GUARDRAIL 1: No chunks at all → no LLM call
if chunk_count == 0:
    logger.warning("[%s] No chunks retrieved — skipping LLM, returning empty result", _qid)
    return {
        "answer": (
            "📭 *No documents found in the index.*\n\n"
            "The index is empty — either parsing failed or no docs were uploaded.\n"
            "Check that `ebooklib` + `html2text` are installed for EPUB support.\n"
            "Try: `uv sync --extra epub` or re-upload documents."
        ),
        "sources": [],
    }

# GUARDRAIL 2: Token budget check
total_input_tokens = sum(len(t.split()) * 1.3 for t in chunk_texts)
system_prompt_tokens = 200  # rough estimate for LlamaIndex overhead
available_for_response = 2048 - total_input_tokens - system_prompt_tokens

if total_input_tokens > 1800:  # too close to context limit
    logger.warning("[%s] Token budget exceeded: ~%d input + %d system > 2048 ctx",
                   _qid, int(total_input_tokens), system_prompt_tokens)
    # Truncate chunks to fit budget
    # or return early with warning

if available_for_response < 50:
    logger.error("[%s] Insufficient context headroom: only %d tokens for response",
                 _qid, int(available_for_response))
    return {
        "answer": "⚠️ *Context window too full.* The retrieved chunks are too large. Try a more specific question.",
        "sources": sources,
    }
```

## Layer 2: Index Health Check (engine.py + indexer.py)

Before retrieval, verify index is usable:

```python
# In query_with_sources() — before Phase 1
index = _index  # set by study handler

try:
    doc_count = len(index.docstore.docs)
    if doc_count == 0:
        logger.warning("[%s] Index has 0 documents!", _qid)
        return {
            "answer": "📭 *Index is empty.* Run `/index` to rebuild it.",
            "sources": [],
        }
except Exception as e:
    logger.error("[%s] Index health check failed: %s", _qid, e)
    # Trigger async re-index instead of crashing
```

And in `DocumentIndexer.ensure_index()`:

```python
# Check if stored index is corrupt
stored_path = os.path.join(INDEX_STORAGE_DIR, "index.json")
if os.path.exists(stored_path):
    try:
        with open(stored_path) as f:
            data = json.load(f)
        if not data:  # empty JSON
            logger.warning("Stored index is empty — will rebuild")
            shutil.rmtree(INDEX_STORAGE_DIR, ignore_errors=True)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Stored index corrupt: %s — will rebuild", e)
        shutil.rmtree(INDEX_STORAGE_DIR, ignore_errors=True)
```

## Layer 3: Modelfile Hardening (Modelfile.3b)

Explicit VRAM and CPU limits so Ollama can't oversubscribe:

```dockerfile
# Modelfile.3b — already has num_ctx 2048 but needs more
PARAMETER num_ctx 1024          # HALVE: halves KV cache → more VRAM headroom
PARAMETER num_gpu 1             # EXPLICIT: only 1 GPU layer (was 99, wrong)
PARAMETER num_thread 4          # Match CPU cores for WSL stability
PARAMETER num_predict 128       # HALVE: shorter responses = faster inference
PARAMETER temperature 0.0
PARAMETER repeat_penalty 1.1    # Reduce hallucination/long generation
PARAMETER stop "Answer:"
PARAMETER stop "<|im_end|>"
```

**`num_gpu 99` was wrong** — that value is for layered models, not Q4_0 quantized. Explicit `num_gpu 1` ensures Ollama uses exactly 1 GPU layer.

## Layer 4: OllamaGuard Enhancement (guard.py)

Add VRAM check before LLM call:

```python
async def ollama_check_vram() -> tuple[bool, str]:
    """Check if we have enough VRAM for inference. Returns (ok, message)."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(",")
            used_mb = float(used.strip())
            total_mb = float(total.strip())
            free_mb = total_mb - used_mb
            
            # We need ~2 GB for model + ~0.5 GB for KV cache
            if free_mb < 1800:
                return False, f"Low VRAM: {int(free_mb)}MB free (need ~2300MB)"
            return True, f"VRAM OK: {int(free_mb)}MB free"
    except Exception as e:
        logger.debug("VRAM check failed: %s", e)
    return True, "VRAM check unavailable"

# In OllamaGuard.__aenter__:
async def __aenter__(self):
    # Existing health check...
    
    # NEW: VRAM check before acquiring lock
    vram_ok, vram_msg = await ollama_check_vram()
    if not vram_ok:
        logger.warning("[%s] Low VRAM: %s", self.operation, vram_msg)
        # Don't block, but log it — we can still try
    
    # ... rest of existing code
```

## Layer 5: TreeSummarize Safety Wrapper (engine.py)

Even if chunks exist, wrap TreeSummarize in a safety timeout with fallback:

```python
# Instead of bare TreeSummarize:
try:
    synthesizer = TreeSummarize(llm=llm)
    response = await asyncio.wait_for(
        synthesizer.aget_response(question, chunk_texts),
        timeout=30,  # HARD CAP: 30s for synthesis
    )
    answer = str(response)
except asyncio.TimeoutError:
    logger.error("[%s] TreeSummarize timed out after 30s", _qid)
    # Fallback: simple join of top chunks
    context = "\n\n".join(chunk_texts[:2])  # Only 2 chunks max
    prompt = f"Based only on this context, answer:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = await llm.acomplete(prompt)
    answer = str(response)
    answer += "\n\n⚠️ *Response was truncated due to timeout. Try a shorter question.*"
except Exception as e:
    logger.error("[%s] TreeSummarize error: %s", _qid, e)
    raise
```

## Layer 6: System Resource Guard (guard.py)

Prevent WSL CPU oversubscription:

```python
# In OllamaGuard.__aenter__ — before Ollama call
# Check system load
try:
    import subprocess
    load = os.getloadavg()[0]  # 1-min average
    cpu_count = os.cpu_count() or 4
    if load > cpu_count * 0.9:  # 90% of cores under load
        logger.warning("System load high: %.1f (cores=%d)", load, cpu_count)
        # Add delay to let system cool
        await asyncio.sleep(2)
except Exception:
    pass
```

## Layer 7: Recovery After WSL Crash (guard.py)

If Ollama is unresponsive after a crash, detect and recover:

```python
# In attempt_ollama_restart():
# Already exists, but add memory check
async def check_ollama_responsive(timeout=3.0) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False

# Before attempting restart, check if it crashed or is just slow
if not await check_ollama_responsive(timeout=3.0):
    # It might be crashed or very slow — give it 5 more seconds
    await asyncio.sleep(5)
    if not await check_ollama_responsive(timeout=10.0):
        logger.error("Ollama unresponsive — likely crashed")
        # Attempt restart
```

---

## Implementation Order

| # | Layer | File | Change |
|---|---|---|---|
| 1 | Pre-LLM validation | `engine.py` | Skip LLM if 0 chunks, token budget check |
| 2 | Index health check | `engine.py` + `indexer.py` | Validate index before retrieval |
| 3 | Modelfile hardening | `Modelfile.3b` | `num_ctx=1024`, `num_gpu=1`, `num_thread=4`, `num_predict=128` |
| 4 | VRAM check in Guard | `guard.py` | `ollama_check_vram()` before LLM call |
| 5 | TreeSummarize timeout | `engine.py` | 30s hard cap + fallback |
| 6 | System load check | `guard.py` | Load average check before Ollama call |
| 7 | ebooklib + html2text | `pyproject.toml` | Add to dependencies, `uv sync` |

---

## What This Prevents

| Failure Mode | Guardrail That Catches It |
|---|---|
| Empty index → bare TreeSummarize | Layer 1: 0-chunk early return |
| Huge chunks → token overflow | Layer 1: token budget check |
| Corrupt index JSON → crash | Layer 2: index health check |
| VRAM exhausted → WSL crash | Layer 3: `num_gpu=1` + Layer 4: VRAM pre-check |
| TreeSummarize hangs forever | Layer 5: 30s timeout + fallback |
| CPU oversubscription → WSL freeze | Layer 6: system load check |
| EPUB parsing always fails | Layer 7: `ebooklib` installed |

---

## Decision Needed

**Execute all layers at once (P1-P7) or stagger?**

Recommend: **All at once** — they're independent and layered. Test after all applied.