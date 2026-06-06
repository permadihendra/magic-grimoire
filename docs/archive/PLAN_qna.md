# Magic Grimoire — `/qna` Command Plan

> **Status:** Proposed · **Priority:** High  
> Creates comprehension Q&A pairs from documents.

---

## What `/qna` Does

A slash command that generates **question + answer pairs** to test comprehension of a topic from the user's documents. Unlike `/quiz` (which can be open-ended), `/qna` always produces explicit Q&A pairs.

```
/qna <topic> [count]
/qna krishna teachings
/qna mahabharata 15
```

## Default Count Analysis

Each Q&A pair = question (~30 tokens) + answer (~80 tokens) = **~110 tokens**

With `num_ctx=4096` and synthesis timeout **60s**:

| Count | Output tokens | Gen time (25 tok/s) | Fits 60s? | Fits 4096 ctx? |
|---|---|---|---|---|
| 5 | 550 | 22s | ✅ | ✅ |
| **10** | **1,100** | **44s** | **✅** | **✅** |
| 12 | 1,320 | 53s | ✅ | ✅ |
| 15 | 1,650 | 66s | ❌ 60s limit | ✅ |
| 20 | 2,200 | 88s | ❌ | ⚠️ 3,100 total |

**Recommended default: 10 pairs.** Gives comprehensive coverage, fits within 60s synthesis timeout, leaves room for context window.

---

## Complement-on-Rerun Logic

If user calls `/qna krishna teachings` twice, the second call should NOT repeat the same questions — it should complement with new ones.

### Mechanism

In-memory dict (survives within session, lost on restart):
```python
_qna_history: dict[tuple[int, str], list[dict]] = {}
# key: (chat_id, topic_lower)
# value: [{"q": "What is karma?", "a": "Action..."}, ...]
```

**First call:**
1. Retrieve chunks from index on topic
2. Generate 10 Q&A pairs
3. Store pairs in `_qna_history[(chat_id, topic)]`
4. Return formatted list

**Second call (same topic):**
1. Retrieve previous pairs from `_qna_history`
2. Include them in prompt as "already covered" context
3. Prompt instructs: "Generate {count} NEW questions that complement the existing ones below. Do NOT repeat or rephrase any covered topics."
4. Append new pairs to history
5. Return full list (old + new, labeled as Part 1, Part 2)

### Prompt Template

```
You are an exam preparation tutor. Based on the provided context
from the user's study documents, generate {count} question + answer
pairs for comprehension testing.

{existing_pairs_block}  ← only on rerun

For each question:
Q: [Clear, specific question about the material]
A: [Complete answer based ONLY on the provided context]

{instructions_block}
```

Where `existing_pairs_block` is:
```
PREVIOUSLY GENERATED QUESTIONS (already answered — do NOT repeat):
1. What is karma?
   Answer: Action and its consequences...
2. Who is Arjuna?
   Answer: The Pandava prince...

Generate {count} NEW questions that complement these.
```

---

## Message Format

```
📝 *Q&A: Krishna Teachings* (10 pairs)

[1] Q: What is the concept of dharma in the Bhagavad Gita?
    A: Dharma refers to righteous duty... (2-3 sentences)

[2] Q: How does Krishna define karma yoga?
    A: Karma yoga is the path of selfless action...

[3-10...]

💡 *Next steps:* Want more questions on this topic?
Just send `/qna krishna teachings` again for new questions!

📊 *Retrieval:* 5 chunks | ~800 tokens | k=3
```

If called twice:
```
📝 *Q&A: Krishna Teachings* (Part 1 — first 10 pairs)
[1-10...]

📝 *Q&A: Krishna Teachings* (Part 2 — new 10 pairs)
[11-20...]
```

---

## Files to Create/Change

| File | Change |
|---|---|
| `app/rag/prompts.py` | Add `QNA_BASE`, `QNA_PROMPT` template with existing-pairs context |
| `app/plugins/study/handler.py` | Add `_qna_history` dict, add `_handle_qna()` method, register `/qna` command |
| `app/plugins/system/handler.py` | Add `/qna` to `/help` output |

---

## Summary

| Aspect | Decision |
|---|---|
| Default count | **10** (550-1,100 tokens, fits 60s timeout, fits 4096 ctx) |
| Duplicate prevention | **In-memory dict** `_qna_history[(chat_id, topic)]` — prompt includes previous pairs with "do NOT repeat" instruction |
| Rerun behavior | Generates new pairs, appends to history, returns as "Part 2" |
| Storage | In-memory only (lost on restart — acceptable for a learning tool) |
| Generation path | Uses existing `engine.query_with_sources()` with `mode="qa"` + QnA-specific prompt |

Ready to execute?
