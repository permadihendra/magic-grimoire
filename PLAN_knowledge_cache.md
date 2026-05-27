# Magic Grimoire — PLAN: Knowledge Cache v2 (Q&A Pairs + Gemini Judge)

## Architecture

```
User: "key takeaways from world economy"
  ↓
┌─ Step 1: Fast Cosine Lookup (50ms, no Gemini) ────────┐
│  Embed query → cosine against stored Q&A embeddings    │
│  Top-3 candidates with similarity > 0.85               │
└────────────────────────────────────────────────────────┘
  ↓
┌─ Step 2: Gemini Judges (100ms) ───────────────────────┐
│  "User asks: [question]                                │
│   Cached:                                              │
│   1. Q: 'top takeaways world economy' A: 'The key...'  │
│      Similarity: 0.95 | Source: World Economy          │
│                                                        │
│   Does any cached answer match?                        │
│   If YES → respond with the cached answer              │
│   If NO  → TOOL: ask(query=..., document=...)"         │
└────────────────────────────────────────────────────────┘
  ↓
  ├─ Cache HIT:  Gemini returns answer (150ms total) ✅
  │              Increment used_count
  │
  └─ Cache MISS: Gemini calls TOOL: ask()
                 → LlamaIndex (10-30s)
                 → Store new Q&A pair in cache
```

## Why Gemini as Judge?

| Without Gemini | With Gemini |
|---|---|
| Hardcoded 0.92 threshold | Gemini decides if answer actually matches intent |
| "loss aversion" (Finance) matches "loss aversion" (World Economy) at 0.90 → cached ❌ | Gemini sees: cached source=Finance, user wants World Economy → "NO — different document. TOOL: ask()" ✅ |
| No ability to partially match | "The cached answer covers part of this. Let me add what's missing. TOOL: ask()" |
| Threshold requires tuning | Zero tuning. Gemini understands nuance |

## Gemini Prompt (Injected into Existing AGENT_PROMPT)

The cache context is ADDED to the existing agent prompt only when candidates exist:

```python
cache_context = f"""
KNOWLEDGE CACHE:
These Q&A pairs were previously answered and approved. Check if any FULLY answers 
the user's question. Consider:
- Is the question asking the same thing? (similar intent, not just similar words)
- Is the answer from the RIGHT document? (if user mentions a book, cached answer must be from same book)
- Is the answer still VALID? (documents may have been re-indexed since)

Cached pairs (best match first):
{candidates_formatted}

DECISION:
- If a cached answer FULLY matches → respond with it directly. You can polish wording slightly.
  Add "💾 Retrieved from previous session" at the end.
- If PARTIALLY matches → use the cached answer AND call TOOL: ask() for what's missing.
- If NO match → just call TOOL: ask() as normal. Don't mention the cache.

User's question: {message}
"""
```

**Important: Only inject when top candidate > 0.85 similarity.** Below that, skip the cache check entirely — save tokens.

## Q&A Pair Schema

```sql
CREATE TABLE knowledge_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,              -- original user question
    question_embedding BLOB,             -- 768 float32 (nomic-embed-text)
    short_answer TEXT NOT NULL,          -- concise answer (first 500 chars)
    full_answer TEXT NOT NULL,           -- complete answer with sources
    source_doc TEXT,                     -- display_name of source document
    retrieval_score REAL,                -- original LlamaIndex confidence
    approved INTEGER DEFAULT 0,          -- 0=auto-cached, 1=user-approved
    used_count INTEGER DEFAULT 0,        -- reuse counter
    chat_id INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME
);

CREATE INDEX idx_pairs_created ON knowledge_pairs(created_at);
CREATE INDEX idx_pairs_approved ON knowledge_pairs(approved);
```

## When Pairs Are Stored

| Trigger | approved | TTL |
|---|---|---|
| Answer generated successfully (auto) | 0 | Survives until /index or 7 days |
| User says "good"/"perfect" (feedback) | 1 | Survives /index (user trusts it) |
| User says "wrong source" | NOT stored | Bad data stays out |
| `/index` runs | approved=0 deleted | approved=1 kept (user vetted) |

## Caching the Answer Text

Two fields:
- `short_answer` (500 chars) — sent to Gemini for cache evaluation (saves tokens)
- `full_answer` (unlimited) — returned to user on cache hit

Gemini only sees short_answer in the evaluation prompt. On cache hit, full_answer is returned to user.

## Integration with BrainPlugin

```python
async def handle(self, ctx: BotContext):
    message = ctx.message_text.strip()
    chat_id = ctx.chat_id
    
    # Step 1: Check cache for high-similarity candidates
    cache_candidates = await check_cache(message, top_k=3, min_sim=0.85)
    
    # Step 2: Build agent prompt
    if cache_candidates:
        history_block = _build_history_block(chat_id)
        cache_block = _build_cache_context(cache_candidates)
        full_prompt = AGENT_PROMPT + "\n\n" + cache_block
        user_message = f"{history_block}\nCurrent message: {message}" if history_block else message
    else:
        # Normal flow — no cache candidates worth evaluating
        history_block = _build_history_block(chat_id)
        full_prompt = AGENT_PROMPT
        user_message = f"{history_block}\nCurrent message: {message}" if history_block else message
    
    # Step 3: Send to Gemini (as before)
    response = await gemini_chat(
        system_prompt=full_prompt,
        user_message=user_message,
    )
    
    # Step 4: If Gemini returned a cached answer (no TOOL line)
    if "TOOL:" not in response:
        # Cache hit! Update used_count
        await increment_used_count(cache_candidates[0]['id'])
        _remember(chat_id, message, response)
        return response
    
    # Step 5: Normal slow-path (TOOL: ask, quiz, etc.)
    ...
    
    # Step 6: After successful ask(), store in cache
    await store_pair(question, answer, source_doc, approved=0)
```

## Token Cost of Cache Context

```python
def _build_cache_context(candidates):
    lines = ["KNOWLEDGE CACHE:"]
    for i, c in enumerate(candidates[:3]):
        lines.append(
            f"{i+1}. Q: {c['question'][:100]}\n"
            f"   A: {c['short_answer'][:200]}\n"
            f"   Similarity: {c['similarity']:.2f} | "
            f"Source: {c['source_doc'] or 'unknown'} | "
            f"Approved: {'yes' if c['approved'] else 'no'}"
        )
    lines.append("\nDECISION: ...")
    return "\n".join(lines)
```

~300 tokens for 3 candidates. Added only when top candidate > 0.85 similarity. 
For 90% of queries (no cached match), zero token overhead.

## Latency Breakdown

| Path | Steps | Time |
|---|---|---|
| **Cache HIT** | Embed (50ms) + cosine (1ms) + Gemini (100ms) | **~150ms** |
| **Cache MISS** | Embed + cosine + Gemini → TOOL: ask() → LlamaIndex | **10-30s** |
| **No candidates** | Skip cache entirely → Gemini → TOOL: ask() → LlamaIndex | **10-30s** |
| **Current (no cache)** | Gemini → TOOL: ask() → LlamaIndex | **10-30s** |

Cache miss adds ~150ms overhead — invisible compared to 10-30s generation.

## Files

| File | Change |
|---|---|
| `app/rag/knowledge_cache.py` | **NEW** — search_cache(), store_pair(), embedding helpers |
| `app/database.py` | Add knowledge_pairs table |
| `app/plugins/brain/handler.py` | Inject cache context into agent prompt, store pairs after ask |
| `app/plugins/study/handler.py` | Invalidate auto-cached pairs on /index |

## Risk: Gemini Hallucinates from Cache

What if Gemini sees a cached Q&A and "remixes" it incorrectly — combining parts of the cached answer with its own knowledge?

**Mitigation:** Prompt rule: "If using a cached answer, reproduce it VERBATIM. Only fix typos or formatting. Do not add new information."

## Verdict

✅ **Better than pure cosine threshold.** Gemini handles nuance: same question different document, partially matching intents, outdated answers. The 150ms overhead on cache miss is invisible. The 100x speedup on cache hit is transformative.

This is the knowledge layer you asked for — Q&A pairs, Gemini-judged, zero threshold tuning.
