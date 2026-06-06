# Magic Grimoire — RAG v2: Detailed Implementation Plan

> **Status:** Approved · **Created:** 2026-06-06  
> **Goal:** Fix ~80% retrieval failure rate. Target: >80% precision (correct doc in top-5).

---

## Executive Summary

The current RAG pipeline retrieves 3 chunks via vector similarity, then synthesizes an answer.
The problem: vector similarity alone is imprecise — wrong chunks, wrong docs, missed context.

**RAG v2 adds three stages between retrieval and synthesis:**
1. **Query Refiner** — Gemini cleans the query + generates keyword variations
2. **Cross-Encoder Reranker** — Local model re-scores 20 chunks → top 5
3. **BM25 Fallback** — Keyword search when rerank returns nothing relevant

**What stays the same:** LLM synthesis (TreeSummarize), source citations, guardrails, OllamaGuard.

---

## Verified Environment

| Component | Status | Notes |
|---|---|---|
| `sentence-transformers` 5.5.1 | ✅ Installed | `CrossEncoder` available |
| `torch` 2.12.0+cu130 | ✅ CUDA available | RTX 3050 |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | ✅ Tested on GPU | ~210MB VRAM, batch predict works |
| Ollama 0.23.4 | ⚠️ No rerank API | Must use sentence-transformers directly |
| `rank_bm25` | ❌ Not installed | Needs `pip install rank_bm25` |

**VRAM after loading reranker:** 1224MB (was 1014MB idle) — **210MB for reranker**.

---

## VRAM Budget (Final)

```
Component                          VRAM
─────────────────────────────────────────
LLM (magic-grimoire:3b)           1.9 GB
KV cache (4096 ctx)               ~2.4 GB
Embedding (nomic-embed-text)      ~0.3 GB
Cross-encoder (ms-marco)          ~0.2 GB
─────────────────────────────────────────
Total                             ~4.8 GB
Headroom                          ~3.2 GB   ← enough for inference spikes
```

---

## New Pipeline

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│  ① Query Refiner (Gemini)                │  ~50ms
│  - Strip conversational filler           │
│  - Generate 3-5 keyword variations       │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  ② Over-Retrieve (LlamaIndex)            │  ~200-400ms
│  - Top 20 chunks (was 3)                 │
│  - For each query variation              │
│  - Merge + deduplicate                   │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  ③ Cross-Encoder Reranker (local GPU)    │  ~300-600ms
│  - Re-score 20 chunks against query      │
│  - Filter: score < threshold → discard   │
│  - Output: top 5 chunks                  │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  ④ Existing Pipeline (unchanged)         │  ~2-4s
│  - Document boosting + feedback          │
│  - TreeSummarize synthesis               │
│  - Source citations + diagnostics        │
└─────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│  ⑤ BM25 Fallback (if ③ returns 0)       │  ~100ms
│  - Keyword search on raw chunk text      │
│  - If still empty → honest "not found"   │
└─────────────────────────────────────────┘
```

**Estimated total latency: 3-6s** (was 2-4s, +1-2s for refine+rerank).

---

## Step-by-Step Implementation

### Step 1: Add `rank_bm25` dependency

**File:** `pyproject.toml`

```toml
dependencies = [
    ...,
    "rank-bm25>=0.2",
]
```

```bash
uv sync
```

---

### Step 2: Add RAG v2 config settings

**File:** `app/config.py`

Add after `retrieval_top_k`:

```python
# ── RAG v2: Query Refinement + Reranking ────────────────
use_query_refiner: bool = True          # Gemini query refinement
use_reranker: bool = True               # Cross-encoder reranking
use_bm25_fallback: bool = True          # BM25 if rerank fails

retrieval_top_k: int = 20               # Over-retrieve (was 3)
reranker_top_k: int = 5                 # After rerank
reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
reranker_threshold: float = 0.0         # Min score to keep (ms-marco uses logit scores)
reranker_device: str = "cuda"           # "cuda" or "cpu" (fallback)
reranker_timeout: float = 10.0          # Seconds before CPU fallback

# Legacy: keep old setting name for backward compat
# retrieval_top_k is now 20 (was 3) — used by over-retrieve
```

**Note on `reranker_threshold`:** `ms-marco-MiniLM-L6-v2` outputs logits (not probabilities). Positive = relevant, negative = irrelevant. Threshold of 0.0 means "keep chunks with positive relevance score." Can be tuned after testing.

---

### Step 3: Create `app/rag/reranker.py` — Cross-Encoder Reranker

**New file.**

```python
"""Cross-encoder reranker — re-score retrieved chunks for relevance.

Uses sentence-transformers CrossEncoder on GPU (fallback to CPU).
Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~80MB, fast on GPU).
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_model = None
_model_device = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker model."""
    global _model, _model_device
    if _model is not None:
        return _model, _model_device

    from app.config import settings
    import torch

    device = settings.reranker_device
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model: %s on %s", settings.reranker_model, device)
        _model = CrossEncoder(
            settings.reranker_model,
            device=device,
            max_length=512,
        )
        _model_device = device
        logger.info("Reranker loaded on %s", device)
        return _model, _model_device
    except Exception as e:
        logger.warning("Failed to load reranker on %s: %s — trying CPU", device, e)
        try:
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(
                settings.reranker_model,
                device="cpu",
                max_length=512,
            )
            _model_device = "cpu"
            logger.info("Reranker loaded on CPU (fallback)")
            return _model, _model_device
        except Exception as e2:
            logger.error("Reranker load failed completely: %s", e2)
            return None, None


def rerank_chunks(
    query: str,
    chunks: list[dict],  # [{"text": ..., "filename": ..., "score": ...}, ...]
    top_k: int | None = None,
) -> list[dict]:
    """Re-score chunks using cross-encoder and return top_k.

    Args:
        query: The search query.
        chunks: List of chunk dicts with 'text', 'filename', 'score' keys.
        top_k: Number of chunks to return (default: settings.reranker_top_k).

    Returns:
        Re-sorted list of chunks with added 'rerank_score' key.
        Empty list if reranker unavailable.
    """
    from app.config import settings

    if top_k is None:
        top_k = settings.reranker_top_k

    if not chunks:
        return []

    model, device = _get_reranker()
    if model is None:
        logger.warning("Reranker unavailable — returning original chunks")
        return chunks[:top_k]

    try:
        t0 = time.time()

        # Build (query, document) pairs
        pairs = [(query, c["text"]) for c in chunks]

        # Predict scores
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank score descending
        reranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)

        # Filter by threshold
        threshold = settings.reranker_threshold
        filtered = [c for c in reranked if c.get("rerank_score", 0) >= threshold]

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "Reranker: %d → %d chunks (threshold=%.2f, %.0fms, device=%s)",
            len(chunks), len(filtered), threshold, elapsed_ms, device,
        )

        return filtered[:top_k]

    except Exception as e:
        logger.error("Reranker failed: %s — returning original chunks", e)
        return chunks[:top_k]


def rerank_available() -> bool:
    """Check if reranker model is loaded and available."""
    model, _ = _get_reranker()
    return model is not None
```

**Key design decisions:**
- Lazy loading — model loads on first rerank call, not at startup
- CPU fallback — if GPU OOM, falls back to CPU automatically
- No OllamaGuard needed — this doesn't use Ollama, it uses sentence-transformers directly
- Returns original chunks if reranker fails — graceful degradation

---

### Step 4: Create `app/rag/query_refiner.py` — Gemini Query Refinement

**New file.**

```python
"""Query refiner — transform raw user questions into optimal search queries.

Uses Gemini to strip filler and generate keyword variations.
Lightweight (~50ms), always runs before retrieval.
"""

import logging
import re

logger = logging.getLogger(__name__)

_REFINER_SYSTEM = """You are a search query optimizer. Given a user question, produce 3-5 search queries optimized for vector similarity search.

Rules:
1. Strip conversational filler ("Can you tell me...", "I want to know...", "What does the book say about...")
2. Extract core intent and key terms
3. Generate 3-5 variations with different phrasings
4. Each variation should be 3-8 words
5. Include synonyms and related terms
6. Return ONLY the queries, one per line, no numbering or formatting

Example:
Input: "Can you tell me what the book says about loss aversion in finance?"
Output:
loss aversion finance
loss aversion prospect theory
loss aversion behavioral economics
psychology of financial losses
risk aversion investment decisions"""


async def refine_query(question: str) -> list[str]:
    """Refine a user question into multiple search query variations.

    Args:
        question: Raw user question.

    Returns:
        List of 3-5 refined query strings. Original question is always first.
    """
    from app.config import settings

    if not settings.use_query_refiner:
        return [question]

    try:
        from app.llm.gemini import gemini_chat

        t0 = __import__("time").time()
        response = await gemini_chat(
            system_prompt=_REFINER_SYSTEM,
            user_message=question,
            max_tokens=150,
            timeout=5.0,
        )
        elapsed_ms = (__import__("time").time() - t0) * 1000

        # Parse: split by newlines, clean up
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            line = re.sub(r"^\d+[\.\)]\s*", "", line)  # Remove "1. " prefix
            line = line.strip('"').strip("'")
            if line and len(line) >= 3:
                queries.append(line)

        if not queries:
            logger.warning("Query refiner returned empty — using original")
            return [question]

        # Always include original question as first query
        if question not in queries:
            queries.insert(0, question)

        logger.info("Query refiner: '%s' → %d queries (%.0fms)", question[:40], len(queries), elapsed_ms)
        return queries[:5]  # Max 5 variations

    except Exception as e:
        logger.warning("Query refiner failed: %s — using original", e)
        return [question]
```

**Key design decisions:**
- Gemini only — fast (~50ms), free tier, no local model overhead
- Original question always included — ensures baseline retrieval works
- Max 5 variations — keeps retrieval fast (5 queries × 20 chunks = 100 candidates, deduplicated)
- Graceful fallback — on error, returns original question only

---

### Step 5: Create `app/rag/bm25_fallback.py` — BM25 Keyword Search

**New file.**

```python
"""BM25 fallback search — lexical keyword search when vector search fails.

Used when reranker returns 0 relevant chunks.
Purely local, no API calls, ~100ms.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def bm25_search(
    query: str,
    chunks: list[dict],  # All chunks from index (text, filename, score)
    top_k: int = 5,
) -> list[dict]:
    """Search chunks using BM25 keyword matching.

    Args:
        query: The search query.
        chunks: Full list of chunks from the index.
        top_k: Number of results to return.

    Returns:
        Top-k chunks ranked by BM25 score.
    """
    if not chunks:
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed — skipping BM25 fallback")
        return []

    try:
        t0 = __import__("time").time()

        # Tokenize: simple whitespace + lowercase
        def tokenize(text: str) -> list[str]:
            return re.findall(r"\w+", text.lower())

        corpus = [tokenize(c["text"]) for c in chunks]
        bm25 = BM25Okapi(corpus)

        query_tokens = tokenize(query)
        scores = bm25.get_scores(query_tokens)

        # Attach scores and sort
        for chunk, score in zip(chunks, scores):
            chunk["bm25_score"] = float(score)

        ranked = sorted(chunks, key=lambda c: c.get("bm25_score", 0), reverse=True)

        # Filter: only chunks with positive BM25 score
        filtered = [c for c in ranked if c.get("bm25_score", 0) > 0]

        elapsed_ms = (__import__("time").time() - t0) * 1000
        logger.info("BM25: %d chunks → %d relevant (%.0fms)", len(chunks), len(filtered), elapsed_ms)

        return filtered[:top_k]

    except Exception as e:
        logger.error("BM25 search failed: %s", e)
        return []
```

---

### Step 6: Integrate into `app/rag/engine.py` — The Main Pipeline Change

This is the core change. Modify `RAGEngine.query_with_sources()` to insert the new pipeline stages.

**Changes to `query_with_sources()`:**

#### 6a: Add imports at top of file

```python
# Add after existing imports
from app.rag.query_refiner import refine_query
from app.rag.reranker import rerank_chunks
from app.rag.bm25_fallback import bm25_search
```

#### 6b: Replace Phase 1 retrieval (lines ~280-350)

**Before (current):**
```python
# --- Phase 1: Sync retrieval with document boosting + feedback ---
retriever = self._get_retriever()
nodes = retriever.retrieve(question)
# ... boosting, feedback, filtering ...
```

**After (new):**
```python
# --- Phase 0.5: Query Refinement (Gemini) ---
queries = await refine_query(question)
logger.info("[%s]  [p0.5] Refined to %d queries", _qid, len(queries))

# --- Phase 1: Over-Retrieve with multi-query ---
# Retrieve top_k for each query variation, merge + deduplicate
all_nodes = []
seen_ids = set()

# Temporarily increase retriever top_k for over-retrieve
retriever = self._get_retriever()
original_top_k = retriever._similarity_top_k
retriever._similarity_top_k = settings.retrieval_top_k  # 20

try:
    for q in queries:
        nodes = retriever.retrieve(q)
        for node in nodes:
            node_id = node.node_id if hasattr(node, 'node_id') else node.text[:100]
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                all_nodes.append(node)
finally:
    retriever._similarity_top_k = original_top_k  # Restore

nodes = all_nodes
logger.info("[%s]  [p1] Retrieved %d unique chunks from %d queries", _qid, len(nodes), len(queries))

# ... existing document boosting + feedback adjustments ...
```

#### 6c: Add reranking stage (after boosting, before chunk extraction)

**Insert after the document boosting + feedback section, before `chunk_texts = [n.text ...]`:**

```python
# --- Phase 1.5: Cross-Encoder Reranking ---
if settings.use_reranker and len(nodes) > settings.reranker_top_k:
    # Convert nodes to dicts for reranker
    rerank_input = []
    for node in nodes:
        rerank_input.append({
            "text": node.text or "",
            "filename": node.metadata.get("file_name", "Unknown"),
            "score": float(node.score) if node.score else 0.0,
            "_node": node,  # Keep reference to original node
        })

    # Rerank: 20 → top_k
    reranked = rerank_chunks(question, rerank_input, top_k=settings.reranker_top_k)

    if reranked:
        # Use reranked chunks
        nodes = [r["_node"] for r in reranked if "_node" in r]
        sources = []  # Rebuild sources from reranked
        seen_files = set()
        for r in reranked:
            fname = r.get("filename", "Unknown")
            if fname not in seen_files and len(sources) < _MAX_SOURCES:
                seen_files.add(fname)
                sources.append({
                    "filename": fname,
                    "score": r.get("rerank_score", r.get("score", 0.0)),
                })
        logger.info("[%s]  [p1.5] Reranked: %d → %d chunks", _qid, len(rerank_input), len(nodes))
    else:
        logger.warning("[%s]  [p1.5] Reranker returned empty — trying BM25 fallback", _qid)
```

#### 6d: Add BM25 fallback (after reranking, before LLM call)

**Insert after the reranking section, before the "no source nodes" check:**

```python
# --- Phase 1.7: BM25 Fallback (if rerank returned nothing) ---
if settings.use_bm25_fallback and (not nodes or len(nodes) < 2):
    logger.info("[%s]  [p1.7] Trying BM25 fallback", _qid)

    # Get ALL chunks from index for BM25
    try:
        all_nodes_raw = []
        for doc_id, node in self._index.docstore.docs.items():
            all_nodes_raw.append({
                "text": node.text or "",
                "filename": node.metadata.get("file_name", "Unknown"),
                "score": 0.0,
            })

        bm25_results = bm25_search(question, all_nodes_raw, top_k=settings.reranker_top_k)

        if bm25_results:
            # Convert back to node-like objects (reuse original nodes if possible)
            # For simplicity, use the text directly and rebuild sources
            nodes = []  # Will be rebuilt from bm25_results
            sources = []
            seen_files = set()
            for r in bm25_results:
                fname = r.get("filename", "Unknown")
                if fname not in seen_files and len(sources) < _MAX_SOURCES:
                    seen_files.add(fname)
                    sources.append({
                        "filename": fname,
                        "score": r.get("bm25_score", 0.0),
                    })
            # Set chunk_texts directly from BM25 results
            chunk_texts = [r["text"] for r in bm25_results if r.get("text")]
            logger.info("[%s]  [p1.7] BM25 found %d chunks", _qid, len(chunk_texts))
        else:
            logger.info("[%s]  [p1.7] BM25 found nothing", _qid)
    except Exception as e:
        logger.warning("[%s]  [p1.7] BM25 fallback failed: %s", _qid, e)
```

#### 6e: Update diagnostics footer

**In the footer construction, add reranker info:**

```python
# Add after existing diag line
refine_info = f" | refine={len(queries)}" if len(queries) > 1 else ""
rerank_info = " | reranked" if settings.use_reranker else ""
diag = (
    f"📊 *Retrieval:* {len(chunk_texts)} chunks | "
    f"~{int(total_tokens)} tokens (~{total_words} words) | "
    f"k={settings.retrieval_top_k}{refine_info}{rerank_info} | "
    f"ctx=4096"
)
```

---

### Step 7: Update `_get_retriever()` for over-retrieve

**The retriever is created with `similarity_top_k=settings.retrieval_top_k`.**  
After Step 2, `retrieval_top_k` defaults to 20. But the `_get_retriever()` method caches the retriever.

**Fix:** Don't cache the retriever, or recreate it when top_k changes.

**File:** `app/rag/engine.py`

```python
def _get_retriever(self, top_k: int | None = None) -> VectorIndexRetriever:
    """Get retriever with optional top_k override."""
    if self._index is None:
        raise RuntimeError("No index available")

    k = top_k or settings.retrieval_top_k
    return VectorIndexRetriever(
        index=self._index,
        similarity_top_k=k,
    )
```

Remove the `_retriever` caching — it's cheap to create and avoids stale top_k issues.

---

### Step 8: Update `retrieve_only()` for consistency

**File:** `app/rag/engine.py`

The `retrieve_only()` method is used by quiz/summarize pre-check. It should also benefit from over-retrieve + rerank, but keep it simpler (no query refinement for pre-check).

```python
def retrieve_only(self, question: str, document: str | None = None, chat_id: int | None = None) -> list[dict]:
    """Retrieve passages without LLM generation.

    Uses over-retrieve + optional reranking for better results.
    """
    retriever = self._get_retriever(top_k=settings.retrieval_top_k)  # 20
    nodes = retriever.retrieve(question)

    # ... existing boosting + feedback logic (unchanged) ...

    # Optional reranking
    if settings.use_reranker and len(nodes) > settings.reranker_top_k:
        from app.rag.reranker import rerank_chunks
        rerank_input = [
            {"text": n.text or "", "filename": n.metadata.get("file_name", ""),
             "score": float(n.score) if n.score else 0.0, "_node": n}
            for n in nodes
        ]
        reranked = rerank_chunks(question, rerank_input, top_k=settings.reranker_top_k)
        if reranked:
            nodes = [r["_node"] for r in reranked if "_node" in r]

    # ... existing source extraction (unchanged) ...
    return passages
```

---

## Implementation Order

| Step | File | Change | Risk |
|---|---|---|---|
| 1 | `pyproject.toml` | Add `rank-bm25` dependency | None |
| 2 | `app/config.py` | Add RAG v2 settings | None |
| 3 | `app/rag/reranker.py` | **New file** — cross-encoder reranker | Low |
| 4 | `app/rag/query_refiner.py` | **New file** — Gemini query refinement | Low |
| 5 | `app/rag/bm25_fallback.py` | **New file** — BM25 keyword search | Low |
| 6 | `app/rag/engine.py` | Integrate pipeline (6a-6e) | **Medium** — core change |
| 7 | `app/rag/engine.py` | Fix `_get_retriever()` caching | Low |
| 8 | `app/rag/engine.py` | Update `retrieve_only()` | Low |

**Steps 1-5 are safe** — new files and config, no existing code modified.  
**Step 6 is the riskiest** — modifies the core query pipeline. Must test carefully.

---

## Feature Flags

All new features have on/off switches in `config.py`:

```python
use_query_refiner: bool = True   # Set False to disable Gemini refinement
use_reranker: bool = True        # Set False to disable cross-encoder
use_bm25_fallback: bool = True   # Set False to disable BM25
```

**Rollback strategy:** If anything breaks, set all three to `False` and the system reverts to the old pipeline (vector similarity → TreeSummarize).

---

## Testing Plan

### Test 1: Query Refiner
```bash
# Test Gemini refinement directly
uv run python3 -c "
import asyncio
from app.rag.query_refiner import refine_query
async def test():
    queries = await refine_query('Can you tell me what the book says about karma?')
    print(queries)
asyncio.run(test())
"
```

### Test 2: Reranker
```bash
# Test reranker with known chunks
uv run python3 -c "
from app.rag.reranker import rerank_chunks
chunks = [
    {'text': 'Karma is action and its consequences', 'filename': 'gita.epub', 'score': 0.5},
    {'text': 'The weather is nice today', 'filename': 'notes.txt', 'score': 0.8},
    {'text': 'Dharma refers to righteous duty', 'filename': 'gita.epub', 'score': 0.4},
]
results = rerank_chunks('what is karma', chunks, top_k=2)
for r in results:
    print(f'{r[\"rerank_score\"]:.2f} | {r[\"text\"][:60]}')
"
```

### Test 3: BM25 Fallback
```bash
# Test BM25 with known chunks
uv run python3 -c "
from app.rag.bm25_fallback import bm25_search
chunks = [
    {'text': 'Karma is action and its consequences in Hindu philosophy', 'filename': 'gita.epub', 'score': 0.0},
    {'text': 'The stock market crashed yesterday', 'filename': 'finance.pdf', 'score': 0.0},
]
results = bm25_search('karma dharma', chunks, top_k=2)
for r in results:
    print(f'{r[\"bm25_score\"]:.2f} | {r[\"text\"][:60]}')
"
```

### Test 4: End-to-End
```bash
# Start bot, send test queries
bash app/start-bot.sh

# In Telegram:
/ask what is karma about
/ask explain dharma
/ask what are the pandavas
```

### Test 5: Rollback
```python
# In config.py, disable all v2 features:
use_query_refiner = False
use_reranker = False
use_bm25_fallback = False
# Restart bot — should work exactly as before
```

---

## Success Metrics

| Metric | Before (v1) | Target (v2) | How to Measure |
|---|---|---|---|
| Retrieval precision | ~20% | >80% | Ask 10 questions, check if top-5 chunks answer correctly |
| Fallback trigger rate | ~80% | <10% | Count "no relevant passages" responses |
| Avg query latency | ~3s | <6s | Log `elapsed_ms` in processing dict |
| Reranker VRAM | N/A | <500MB | `nvidia-smi` during query |
| Crash rate | Occasional OOM | Zero | Monitor bot uptime |

---

## Files Changed Summary

| File | Action | Lines Changed |
|---|---|---|
| `pyproject.toml` | Edit | +1 line |
| `app/config.py` | Edit | +12 lines |
| `app/rag/reranker.py` | **Create** | ~90 lines |
| `app/rag/query_refiner.py` | **Create** | ~65 lines |
| `app/rag/bm25_fallback.py` | **Create** | ~55 lines |
| `app/rag/engine.py` | Edit | ~80 lines changed/added |
| **Total** | | ~303 lines |

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Reranker VRAM OOM during LLM inference | Medium | Lazy-load, CPU fallback, feature flag |
| Gemini query refiner adds latency | Low | 50ms typical, timeout=5s, graceful fallback |
| BM25 index not pre-built | Low | Built on-the-fly from docstore (~100ms for 1000 chunks) |
| New pipeline breaks existing queries | Medium | Feature flags — disable all to revert |
| `ms-marco` threshold tuning needed | Low | Start at 0.0, tune after testing |
| Multi-query retrieval returns too many duplicates | Low | Dedup by node_id |
