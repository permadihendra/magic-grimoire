# Magic Grimoire — RAG v2 Implementation Plan

> Status: **Draft** | Branch: `optimization` | Created: 2026-06-06

## Context

After analyzing the system's ~80% retrieval failure rate, two perspectives were evaluated:

1. **Widura (Initial Analysis):** Agentic RAG with Gemini-powered query transformation + Gemini reranking.
2. **Sangkuni (Peer Critique):** Pragmatic approach — local cross-encoder, latency awareness, chunking audit, fallback strategy, disable HyDE.

This plan merges both into the **RAG v2** — a production-ready, local-first pipeline.

---

## The RAG v2 Pipeline

```
User Question
      │
      ▼
┌─────────────────────────────────┐
│  ① Query Refiner (Gemini)        │  ← Lightweight, always runs
│  - De-noise (strip filler)       │
│  - Keyword expansion (3-5 terms)│
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  ② Over-Retrieve (LlamaIndex)    │  ← Top 20 chunks (was top-k)
│  - Vector similarity search      │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  ③ Local Cross-Encoder Reranker │  ← Conditional: only if confidence
│  - BGE-Reranker or ms-marco      │    score of top result < threshold
│  - Filter: Top 20 → Top 5        │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  ④ Synthesis (Local LLM)         │  ← magic-grimoire:3b
│  - TreeSummarize or direct       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│  ⑤ Fallback Guard               │  ← If reranker says "none relevant"
│  - BM25 keyword search           │
│  - If still empty: honest reply  │
└─────────────────────────────────┘
```

---

## Priority Order

| Priority | Component | Why |
|---|---|---|
| 🔴 1 | Local Cross-Encoder Reranker | Biggest impact, zero API cost, fast |
| 🔴 2 | Query Refinement via Gemini | Clean queries = better initial retrieval |
| 🟡 3 | Over-Retrieve then Filter | Pairs with reranker (20 → 5) |
| 🟡 4 | Chunking Strategy Audit | Fix ingestion — garbage in, garbage out |
| 🟡 5 | Fallback Path | "I don't have that" over hallucination |
| 🟢 6 | HyDE | **DISABLED** — too risky on 3B model |

---

## Component Specifications

### 1. Local Cross-Encoder Reranker

**Model:** `BGE-Reranker` or `ms-marco-MiniLM-L6-v2`
**Runtime:** GPU — loaded via Ollama (`ollama pull BAAI/bge-reranker-base`) or sentence-transformers with `device="cuda"`
**VRAM:** ~500MB — fits well within RTX 3050 headroom alongside LLM + embed
**Library:** `sentence-transformers` (GPU-accelerated)

**Behavior:**
- Input: Query + 20 retrieved chunks
- Output: Sorted list of chunks with relevance scores
- Filter threshold: Score < 0.3 → discard
- Output: Top 5 chunks to synthesis

**GPU Loading Strategy (Ollama):**
1. `ollama pull BAAI/bge-reranker-base`
2. Load via Ollama rerank API endpoint (`/rerank`)
3. VRAM managed by Ollama — model stays loaded between queries (fast subsequent calls)

**VRAM Budget (RTX 3050 8GB — all GPU):**
```
LLM (magic-grimoire:3b)     1.9 GB
KV cache (1024 ctx)         ~1.2 GB
Ollama embed (nomic)        ~0.3 GB
Cross-encoder (GPU)          ~0.5 GB
─────────────────────────────────
Total                        ~3.9 GB
Headroom                     ~4.1 GB
```

**Pipeline Timing (GPU-accelerated):**
- Query refinement (Gemini): ~50ms
- Embed + retrieve 20 chunks: ~200-400ms (Ollama embed, GPU)
- Cross-encoder rerank 20→5: ~300-600ms (GPU, Ollama-managed)
- Synthesis (TreeSummarize): ~2-4s (LLM)
- **Total estimated: ~3-5s end-to-end**

**Fallback:** If cross-encoder OOMs, fallback to CPU mode (`device="cpu"`) via sentence-transformers — slower but prevents crash. Ollama VRAM guard handles this automatically.

**Implementation Location:** `app/rag/engine.py`
- Add `RerankerNode` class or integrate into `RAGEngine.query_with_sources()`
- Add `settings.reranker_threshold` and `settings.reranker_ollama_model` to `config.py`
- Add OllamaGuard check before loading reranker model

### 2. Query Refiner (Gemini)

**Role:** Transform raw user question into optimal search queries
**Model:** Gemini 3.1 Flash Lite (existing `app/llm/gemini.py`)
**Latency Target:** < 100ms
**Token Budget:** < 200 tokens input, < 100 tokens output

**Transformation Rules:**
1. Strip conversational filler ("Can you tell me...", "I want to know...")
2. Extract core intent and key terms
3. Generate 3-5 keyword-expanded variations

**Example:**
```
Input:  "Can you tell me what the book says about loss aversion in finance?"
Output: ["loss aversion finance", "loss aversion prospect theory", "loss aversion behavioral economics"]
```

**Implementation Location:** `app/rag/engine.py` → new `QueryRefiner` class
- Called before retrieval in `query_with_sources()`
- Multi-query retrieval: search index with all variations, merge results

### 3. Over-Retrieve then Filter

**Change:** `retrieval_top_k` (currently 5-10) → retrieve 20 chunks
**Filter:** Cross-encoder reranker reduces 20 → 5
**Why:** Vector search is broad but imprecise. Over-retrieve gives the reranker enough candidates to work with.

**Implementation Location:** `app/config.py`
```python
retrieval_top_k: int = 20       # Over-retrieve
reranker_top_k: int = 5         # After rerank
reranker_threshold: float = 0.3 # Min score to keep
```

### 4. Chunking Strategy Audit

**Current State:** Likely fixed-token splitting (e.g., 512 tokens)
**Problem:** Sentences cut in half, context broken
**Goal:** Semantic/chunking that respects sentence and paragraph boundaries

**Options:**
- **Recursive Text Splitting:** Split by headers, paragraphs, then sentences
- **Semantic Chunking:** Split at natural topic boundaries
- **Overlap:** 20% overlap between chunks to preserve context

**Implementation Location:** `app/rag/indexer.py`
- Review `parse_document_multimethod()` chunking logic
- Add `chunk_overlap` and `chunk_size` to settings
- Test with a known document to verify context integrity

### 5. Fallback Path

**Trigger:** Reranker returns 0 chunks above threshold

**Step 1:** BM25 keyword search (purely lexical, no embeddings)
- Use `rank_bm25` or similar library
- Fallback to search index with user keywords

**Step 2:** If still empty, respond:
```
📭 *I searched your library but couldn't find relevant information on this topic.*

Options:
• Try different keywords
• Upload more relevant documents
• Check if this topic is covered in your documents with `/files`
```

**Implementation Location:** `app/rag/engine.py`
- Add `FallbackBM25` class
- Integrate into `query_with_sources()` after reranker

### 6. HyDE — DISABLED

**Reason:** `magic-grimoire:3b` is too small to generate reliable hypothetical answers.
HyDE with a weak generator produces worse retrieval than the raw query.

**Decision:** Remove HyDE from the pipeline entirely for now.
Revisit if the local model is upgraded to >= 7B.

---

## Configuration Changes

**File:** `app/config.py`

```python
# Retrieval settings
retrieval_top_k: int = 20        # Over-retrieve for reranker
reranker_top_k: int = 5          # Final chunks after rerank
reranker_threshold: float = 0.3  # Min score to keep a chunk
use_reranker: bool = True        # Feature flag
use_query_refiner: bool = True   # Feature flag

# Fallback
use_bm25_fallback: bool = True   # Trigger BM25 if rerank fails

# Chunking
chunk_size: int = 512            # Target chunk size in tokens
chunk_overlap: float = 0.2       # 20% overlap between chunks
```

---

## File Changes Summary

| File | Change |
|---|---|
| `app/config.py` | Add reranker, query refiner, chunking, BM25 settings |
| `app/rag/engine.py` | Add QueryRefiner, RerankerNode, BM25 fallback, new pipeline |
| `app/rag/indexer.py` | Audit and fix chunking strategy |
| `app/rag/models.py` | Add cross-encoder model loading |
| `requirements.txt` / `pyproject.toml` | Add `sentence-transformers`, `rank_bm25` |

---

## Testing Plan

1. **Query Refiner Test:** Send same question with/without refinement → compare retrieved chunks
2. **Reranker Test:** Verify 20→5 filtering → check if top 5 actually answer the question
3. **BM25 Fallback Test:** Ask something unrelated to indexed docs → verify fallback triggers
4. **Chunking Test:** Index a known document → verify chunks aren't sentence-split
5. **End-to-End:** Ask 10 questions → measure relevance of final answer

---

## Success Metrics

| Metric | Target |
|---|---|
| Retrieval precision (correct doc in top-5) | > 80% (was ~20%) |
| Average query latency | < 5s end-to-end |
| Cross-encoder GPU VRAM | Stable < 4.5 GB total |
| Fallback trigger rate | < 10% of queries |
| Chunks not sentence-split | 100% |

---

## Notes

- Sangkuni's critique was incorporated. Key changes: cross-encoder replaces Gemini reranker, conditional reranking, chunking audit added, fallback path added, HyDE disabled.
- Parallelize query refinement and retrieval where possible to reduce latency.
- All new models (cross-encoder, BM25) must work on CPU fallback if GPU is busy.