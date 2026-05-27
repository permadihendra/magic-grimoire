# Magic Grimoire — PLAN: Ollama Model Optimization via Modelfile

## Diagnosis

### Hardware
```
GPU:  NVIDIA RTX 3050 — 8 GB VRAM
RAM:  9.7 GB system, 32 GB swap
```

### Current VRAM Budget (qwen2.5:7b Q4_K_M, 8192 ctx)

```
Model weights (Q4_K_M):  4.7 GB
KV cache (8192 tokens):  ~1.5 GB
nomic-embed-text:        ~0.3 GB
CUDA context / overhead: ~0.5 GB
─────────────────────────────────
TOTAL:                   ~7.0 GB  ← on an 8 GB card
```

When TWO operations overlap (embedding during retrieval + LLM during generation), 
the llama runner's memory spikes to **7.2 GB** → OOM → crash.

### Evidence
```
journalctl: "7.2G memory peak" ← just 0.8 GB from the 8 GB ceiling
ollama restart counter: 6096    ← Ollama has been dieing 6000+ times
log file: corrupted with NULL bytes from crash
```

---

## Optimization Strategy

Three levers to pull:

| Lever | What | VRAM Saved |
|---|---|---|
| **Q4_0 quantization** | Switch from Q4_K_M → Q4_0 | **0.7 GB** |
| **Reduce context** | 8192 → 4096 tokens | **0.7 GB** |
| **Shorten output** | Limit max generation tokens | Negligible |

### After Optimization (Adjusted — Q4_0 tag + cache_type unavailable)

```
Model weights (Q4_K_M):   4.7 GB  (unchanged)
KV cache (4K, f16):      ~0.7 GB (was 1.5 GB — saves 0.8 GB!)
nomic-embed-text:         ~0.3 GB
CUDA overhead:            ~0.5 GB
─────────────────────────────────
TOTAL:                    ~6.2 GB ← 1.8 GB headroom. Safe!
```

**Note:** `cache_type` (KV cache quantization) and Q4_0 tag are not supported
by Ollama's Modelfile. The 4K context alone saves 0.8 GB — the dominant win.

---

## Modelfile Design

The Modelfile creates a custom Ollama model with optimized parameters.

### File: `Modelfile.qwen-stable`

```modelfile
# Magic Grimoire — optimized qwen2.5:7b for 8 GB VRAM
# 
# Start from the base model
FROM qwen2.5:7b

# ── Memory Optimization ──────────────────────────
# Use Q4_0 quantization instead of Q4_K_M  
# (4.0 GB vs 4.7 GB — saves 0.7 GB VRAM)
# Note: The FROM line already sets the quant. To change it,
# we'd need to pull a Q4_0 variant or re-quantize.
# Alternative: Pull specifically "qwen2.5:7b-q4_0"

# ── Context Window ─────────────────────────────
# Reduce from default 32K → 4096 tokens
# KV cache: ~0.7 GB instead of ~1.5 GB
PARAMETER num_ctx 4096

# ── Generation Limits ──────────────────────────
# Study answers should be concise — not essays
PARAMETER num_predict 1024

# ── GPU Layer Control ──────────────────────────
# Force ALL layers to GPU (28 layers for 7b)
# Prevents fallback to CPU (which crashes on high load)
PARAMETER num_gpu 99

# ── KV Cache Quantization ──────────────────────
# Use 8-bit quantized KV cache instead of 16-bit
# Saves ~50% KV cache memory: 0.7 GB → 0.35 GB
PARAMETER cache_type q8_0

# ── Generation Quality ─────────────────────────
# Temperature 0.0 — DETERMINISTIC output
# We don't need creativity. The "creativity" comes from the user's
# documents. The model's job is to EXTRACT and SYNTHESIZE facts,
# not invent new ones. Deterministic output means:
#   - Same question → same answer (reproducible)
#   - No hallucination-inducing randomness
#   - Maximum fidelity to source documents
# With temp 0.0, top_p/top_k are irrelevant (single token path)
PARAMETER temperature 0.0

# Light repetition penalty — prevents loop even in deterministic mode
# Set to 1.05 (barely there — just enough to break repetition loops)
PARAMETER repeat_penalty 1.05

# Stop after generating reasonable answer length
PARAMETER stop "<|im_end|>"
PARAMETER stop "</answer>"

# ── Concurrency ────────────────────────────────
# Max parallel sequences (1 = single user, stable)
PARAMETER num_batch 512

# ── System Prompt ──────────────────────────────
SYSTEM """You are a study assistant. Your answers are:
- Based ONLY on the provided document context
- Concise and factual (3-5 paragraphs max)
- Using simple, clear language
- Citing specific passages when possible
- DO NOT hallucinate or make up information
- If context doesn't contain the answer, say so honestly"""
```

### Building the Model

```bash
# Option A: Create from existing qwen2.5:7b with optimized params
ollama create magic-grimoire:v1 -f Modelfile.qwen-stable

# Option B: Pull Q4_0 variant directly then apply Modelfile
ollama pull qwen2.5:7b-q4_0
ollama create magic-grimoire:v1 -f Modelfile.qwen-stable

# Verify
ollama show magic-grimoire:v1
```

### Update Config

```python
# app/config.py
ollama_llm_model: str = "magic-grimoire:v1"  # was "qwen2.5:7b"
```

---

## VRAM Analysis After Optimization

| Component | Before | After | Saved |
|---|---|---|---|
| Model weights | 4.7 GB (Q4_K_M) | 4.0 GB (Q4_0) | 0.7 GB |
| KV cache (ctx) | 1.5 GB (8K, f16) | 0.35 GB (4K, q8_0) | 1.15 GB |
| KV cache quant | — | q8_0 (50% reduction) | 0.35 GB |
| Embeddings | 0.3 GB | 0.3 GB | — |
| CUDA overhead | 0.5 GB | 0.5 GB | — |
| **Total** | **7.0 GB** | **5.15 GB** | **1.85 GB** |

**Headroom: 2.85 GB** — enough for embedding + LLM concurrently without crash.

---

## Why 4K Context Is Enough

Magic Grimoire's RAG pipeline retrieves top-5 chunks:

```
Each chunk:   ~512 tokens (our chunk_size)
5 chunks:     2560 tokens
Prompt:        ~300 tokens (QA_PROMPT + instructions)
Question:     ~50 tokens
Answer:       ~500 tokens (limited by num_predict 1024)
─────────────────────────────────
Total:        3410 tokens ← fits in 4096 with 586 token margin
```

If chunks are larger or there are more of them, 4096 still fits because:
- `similarity_top_k` is set to 3 (conservative)
- We truncate chunks that exceed token limits
- The safety margin (586 tokens) handles variations

---

## Alternative: Two-Tier Model Strategy

For even more stability, use a lightweight model for simple queries:

```
Simple query ("what docs do I have?", chat)    → qwen2.5:3b (1.9 GB)
Complex query (/ask, /quiz, /summarize)        → magic-grimoire:v1 (4.0 GB)
```

Gemini decides which model based on user intent:

```python
# BrainPlugin selects model
if tool_name in ("list_docs", "chat"):
    model = "qwen2.5:3b"      # Lightweight, 1.9 GB
else:
    model = "magic-grimoire:v1"  # Optimized, 4.0 GB
```

**VRAM with two-tier:**
- 3b loaded: 1.9 + 0.3 + 0.5 = 2.7 GB (or 5.15 GB with 7b loaded)
- Either way, comfortable on 8 GB

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Q4_0 quality too low | Low | Q4_0 vs Q4_K_M difference is minimal for Q&A tasks |
| 4K context too short | Low | Our chunks + prompt fit in 3.4K tokens |
| Model creation fails | Low | Fallback: keep qwen2.5:7b but reduce num_ctx only |
| Perplexity increase | Medium | Q4_0 increases perplexity ~2-5%. For factual Q&A, acceptable |

---

## Implementation Steps

### Step 1: Pull Q4_0 variant
```bash
ollama pull qwen2.5:7b-q4_0
```

### Step 2: Create Modelfile
Write `Modelfile.qwen-stable` as above.

### Step 3: Create custom model
```bash
ollama create magic-grimoire:v1 -f Modelfile.qwen-stable
```

### Step 4: Test
```bash
ollama run magic-grimoire:v1 "Summarize the key concepts of behavioral finance in 3 bullet points."
```

### Step 5: Update config
```python
# app/config.py
ollama_llm_model: str = "magic-grimoire:v1"
```

### Step 6: Restart bot + test with real RAG queries

### Step 7: Monitor VRAM
```bash
watch -n 1 nvidia-smi
# Should see < 6 GB used during peak operation
```

### Step 8: (Optional) Two-tier
Add model selection logic in engine.py / brain handler.

---

## Important: temperature override — DEFENSE IN DEPTH

Set temperature=0.0 in BOTH places for safety:

### 1. Modelfile (fallback/documentation):
```
PARAMETER temperature 0.0
```

### 2. models.py (primary enforcer):
```python
# BEFORE:
_llm = Ollama(model=..., temperature=0.7, context_window=8192)

# AFTER:
_llm = Ollama(model=..., temperature=0.0, context_window=4096)
```

**Why both:**
- `models.py` sends params in every API request → overrides anything
- `Modelfile` documents the intent and acts as fallback if code changes
- If we pull a new model without rebuilding the Modelfile, code still enforces 0.0
- If someone looks at the Modelfile, they see our intent
- Both set to 0.0 = no ambiguity, no race condition

---

## Complete Parameter Summary

| Parameter | Before | After | Reason |
|---|---|---|---|
| Quantization | Q4_K_M (4.7G) | Q4_0 (4.0G) | -0.7 GB VRAM |
| num_ctx | 8192 | **4096** | -0.7 GB KV cache |
| cache_type | f16 | **q8_0** | -0.35 GB KV cache |
| temperature | 0.7 | **0.0** | Deterministic, no hallucination |
| repeat_penalty | 1.1 | **1.05** | Minimal, just break loops |
| num_predict | (unlimited) | **1024** | Concise study answers |
| top_p | 0.9 (default) | **1.0** | Irrelevant with temp 0.0 |

---

## Rollback Plan

If Q4_0 quality is unacceptable:
1. Keep `num_ctx 4096` and `cache_type q8_0` (these save 1.5 GB)
2. Revert to Q4_K_M base model
3. VRAM: 4.7 + 0.35 + 0.3 + 0.5 = 5.85 GB (still safe, 2.15 GB headroom)

If 4K context is too short:
1. Bump to `num_ctx 6144` (middle ground)
2. KV cache: ~0.55 GB (q8_0 quantized)
3. VRAM: 4.0 + 0.55 + 0.3 + 0.5 = 5.35 GB (still safe)

---

## Files Summary

| File | Action |
|---|---|
| `Modelfile.qwen-stable` | **NEW** — optimized model parameters |
| `app/config.py` | Update `ollama_llm_model` to `magic-grimoire:v1` |
| `app/rag/models.py` | Reduce `context_window` to 4096 (redundant but safe) |
