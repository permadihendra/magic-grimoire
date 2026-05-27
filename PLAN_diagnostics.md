# Magic Grimoire — PLAN: Fix Gemini Ignoring RAG Tools

## Root Cause

Gemini is **not calling the ask() tool**. It answers from its own training data:

```
User: "whats the bhagavad gita books about?"
Gemini: "I apologize, but it appears that The Bhagavad Gita is not 
        among your currently indexed documents." 
         ← This is Gemini's OPINION, not RAG result
         ← No ask_query logged
         ← No answer generation logged
         ← No source citations shown
```

**Why:** Gemini has a "helpful but lazy" tendency. If it can answer from training data, it will — rather than calling a tool (which costs time/overhead). Our prompt says "ask tool is for documents" but Gemini decides it "knows" the answer.

**Consequence:** User sees no sources, no citations, no diagnostics. Can't evaluate if RAG works.

## Solutions

### Solution 1: Failsafe — Always Show Source Footer (High Impact)

Modify the slow-path: if the final answer does NOT contain "📖 *Primary source:*" or "📎 *Also:*", the answer came from Gemini (not RAG). Append:

```
ℹ️ *This answer came from my training data, not your documents.*
   For document-sourced answers, use `/ask <question>` 
   or say "search your documents for..."
```

This is a **failsafe**: always visible when RAG was skipped. User immediately knows whether the answer is from documents or not.

### Solution 2: Agent Prompt Rewrite (Highest Impact)

The current prompt is too polite. Gemini ignores the "use ask() tool" instruction. Need a STRONGER prompt:

```python
RULES FOR DOCUMENT QUESTIONS:
1. CRITICAL: For ANY question about the content of the user's documents, 
   you MUST call the ask() tool. Do NOT answer from your training data.
   Even if you "know" the answer — the documents are the SOURCE OF TRUTH.
2. If the user asks "what is X in the [document name] book" — 
   ALWAYS call ask(query="X", document="document name").
3. The ask() tool output ALWAYS includes source citations.
   If you see "📖 Primary source:", you used the documents correctly.
4. If you are NOT sure if the question requires documents, 
   call ask() anyway — better to search and find nothing, 
   than to guess from training data.
```

### Solution 3: `/ask` Prefix Bypasses Gemini (Practical)

Currently, `/ask <question>` goes directly to StudyPlugin (bypasses Gemini). This should ALWAYS work and ALWAYS show sources:

```
/ask what is the bhagavad gita about?
  → StudyPlugin._handle_ask() → RAG Engine → answer + sources ✅
  → No Gemini involvement — guaranteed document search
```

User can always fall back to `/ask` when they want certainty.

### Solution 4: Detection Footer (Quick Win)

In the slow-path, after tool execution, detect if any tool was actually called:

```python
if not any("TOOL:" in line for line in result_lines):
    # No tool was called — Gemini answered from its own knowledge
    final_reply += (
        "\n\nℹ️ *This answer came from my training data.* "
        "For document-sourced answers, preface with `/ask`."
    )
```

## Implementation Plan

### Phase 1: Agent Prompt + Detection Footer (Quick)

| File | Change |
|---|---|
| `app/plugins/brain/handler.py` | Add detection footer to slow-path (Solution 4) |
| `app/plugins/brain/handler.py` | Strengthen agent prompt (Solution 2) |

### Phase 2: Diagnostics in Response (After Retrieval Works)

| File | Change |
|---|---|
| `app/rag/engine.py` | Add 📊 diagnostics footer to query_with_sources() |
| `app/rag/engine.py` | Add per-chunk verbose mode |

### Phase 3: Evaluation

After these changes:
1. User sees "ℹ️ This answer came from training data" when Gemini skipped RAG
2. User sees "📊 Retrieval: 3 chunks | ~890 tokens" when RAG was used
3. User can immediately tell if the answer is document-grounded or not
