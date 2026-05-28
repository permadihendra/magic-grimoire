# Magic Grimoire — Gemini-Powered QnA Plan

> **Status:** Proposed  
> `/qna pandavas` returns "photosynthesis" → local 3B model can't follow structured QnA prompts reliably

---

## Root Cause

`/qna` uses the local Ollama LLM (magic-grimoire:3b) for generation via TreeSummarize. The 3B model struggles to follow the QNA_PROMPT's structured format, especially for topics with sparse or ambiguous document coverage. It falls back to its training data (photosynthesis) instead of using the retrieved document context.

---

## Fix: Route QnA Generation Through Gemini

### Current flow

```
/qna pandavas
  → retrieve chunks (Ollama embeddings, ✓ fine)
  → TreeSummarize + QNA_PROMPT (Ollama LLM, ✗ hallucinates)
  → return Q&A pairs
```

### Proposed flow

```
/qna pandavas
  → retrieve chunks (Ollama embeddings, ✓ same as before)
  → send chunks + QNA_PROMPT to Gemini (✓ fast, accurate, 32K context)
  → Gemini returns structured Q&A pairs
  → return result
```

### Why Gemini is better for this

| Factor | Ollama 3B | Gemini Flash Lite |
|---|---|---|
| Following structured prompts | ❌ Often drifts | ✅ Excellent |
| Context window | 2048-4096 | 32K |
| Hallucination with sparse docs | ❌ Falls back to training data | ✅ Better at saying "not in docs" |
| Speed | ~40 tok/s (50s for 10 Q&A) | ~2-5s for full response |
| Cost per query | Free (local) | Free tier |
| Timeout issues | ❌ 120s timeout needed | ✅ Rarely hangs |

### Implementation

**Change `_handle_qna()`** to:
1. Pre-retrieve passages (already done)
2. Build a Gemini prompt with the QNA_PROMPT + passages + topic
3. Call `gemini_chat()` instead of `_rag_engine.query_with_sources()`
4. Parse and return Gemini's response

```python
async def _handle_qna(self, ctx: BotContext):
    # 1. Retrieve passages (same as before)
    preview_passages = _rag_engine.retrieve_only(topic)
    
    # 2. Build Gemini prompt
    context_text = "\n\n".join(p["text"] for p in preview_passages[:5])
    personality = settings.ai_personality.strip()
    gemini_prompt = f"""Based on the following passages from the user's documents, generate {count} question + answer pairs for comprehension testing about "{topic}".

Passages:
{context_text}

Generate {count} Q&A pairs. Format each as:
[1] Q: [question]
    A: [answer 2-4 sentences]
..."""
    
    # 3. Call Gemini
    response = await gemini_chat(gemini_prompt, chat_id=ctx.chat_id)
    
    # 4. Return formatted result
    answer_text = response.strip()
    ...
```

### Only for slash command `/qna`

The Gemini tool (`_tool_qna` → `generate_qna()`) keeps using Ollama as before. The slash command is the main user-facing path that needs fixing. (Or I can update both if the first approach works.)

### Keep `_rag_engine.retrieve_only()` for retrieval

Embeddings still run via Ollama — that's fast, cheap, and was never the problem. Only generation switches to Gemini.

---

## Files Changed

| File | Change |
|---|---|
| `app/plugins/study/handler.py` | Replace `query_with_sources(mode="qna")` with Gemini-based generation in `_handle_qna()` |

Ready to execute?
