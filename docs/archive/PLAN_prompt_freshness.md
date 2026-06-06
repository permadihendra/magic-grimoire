# Magic Grimoire — Prompt Freshness & Variation Plan

> **Status:** Proposed  
> Same question → same answer. Even without the cache, `temperature=0.0` makes output deterministic for identical input.

---

## Root Cause

Two factors combine to produce identical answers for repeated questions:

### Factor 1: Knowledge cache (already planned for removal)
Returns cached result without calling the LLM.

### Factor 2: `temperature=0.0`
After cache removal, the LLM still produces the **exact same output** for the same prompt + context. Temperature 0.0 means greedy decoding — always pick the most likely token. No randomness at all.

```
Ask "what is karma" → prompt A + context A + temp 0.0 → output A
Ask "what is karma" → prompt A + context A + temp 0.0 → output A (identical)
```

### Factor 3: Prompt templates don't encourage variety
The prompts (QA_NORMAL, QUIZ_NORMAL, SUMMARY_PROMPT, QNA_PROMPT) give static instructions. No hint that the answer should vary between sessions.

---

## Fix: Three Changes

### Change 1: Remove knowledge cache (already planned)
Remove cache check and cache store from `engine.py`. Fresh generation every time.

### Change 2: temperature 0.0 → 0.1
Tiny increase — barely noticeable for factual answers, but enough to produce varied sentence structures and examples for repeated questions.

At 0.1:
- The most likely token still gets picked ~99% of the time
- Only ties between near-equal tokens break differently
- Answer content stays accurate (constrained by document context)
- Wording, examples, and structure vary slightly

**Files:** `Modelfile.3b:17` and `models.py:43`

### Change 3: Add "fresh perspective" to all prompt templates
Add a single line to each prompt asking for variation. The model doesn't know what was generated before, so this acts as a general "be creative" signal.

**QA_NORMAL:**
```
+ - Each time you answer, approach the topic from a slightly different angle.
+   Vary your examples, structure, and emphasis.
```

**QUIZ_NORMAL:**
```
+ - Vary the questions between sessions — change scenarios, examples, and emphasis.
+   Don't repeat the exact same questions.
```

**SUMMARY_PROMPT:**
```
+ - Each time you summarize, highlight different aspects or use a different
+   organizational structure.
```

**QNA_PROMPT:**
```
+ - Vary the questions between sessions. Use different scenarios, examples,
+   and emphasis. Don't repeat the same Q&A pairs.
```

---

## Files Changed

| File | Change | Why |
|---|---|---|
| `app/rag/engine.py:276-281` | Remove cache check | Stale identical answers |
| `app/rag/engine.py:557-565` | Remove cache store | No cache = no stale answers |
| `app/plugins/study/handler.py` | Remove `_qna_history` | Fresh QnA every time |
| `app/rag/models.py:43` | `temperature=0.0` → `0.1` | Tiny randomness for variation |
| `Modelfile.3b:17` | `PARAMETER temperature 0.0` → `0.1` | Match models.py |
| `app/rag/prompts.py` | Add "vary between sessions" to QA, Quiz, Summary, QnA prompts | Explicit instruction for diversity |

---

## Expected Behavior After Fix

```
/ask what is karma → Answer A (fresh LLM call)
/ask what is karma → Answer B (different examples, slightly different structure)
/qna krishna teachings → Q&A set A
/qna krishna teachings → Q&A set B (different questions)
```

Content stays factual (constrained by document context). But wording, examples, and question selection vary naturally.

Ready to execute?
