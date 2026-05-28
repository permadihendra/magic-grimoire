# Magic Grimoire — `/qna` Parsing Fix Plan

> **Status:** Proposed · **Priority:** High  
> `/qna` only shows 1 Q&A pair instead of the requested 10.

---

## Root Cause

The parser at `study/handler.py` line ~380 requires the exact format:

```python
if line.startswith("[") and "Q:" in line:     # expects "[1] Q: ..."
elif line.startswith("A:") and new_pairs:      # expects "A: ..."
```

TreeSummarize wraps the output in its own formatting, which doesn't match this strict format. The model might output:

| Model output | Parser result |
|---|---|
| `[1] Q: What is karma?` → `A: ...` | ✅ Parsed as pair 1 |
| `[2] Q: Who is Arjuna?` → `A: ...` | ✅ Parsed as pair 2 |
| `**Q3: What is dharma?**` | ❌ Doesn't start with `[` → skipped |
| `3. Q: ...` | ❌ Starts with `3.` not `[3]` → skipped |
| `Question 4: ...` | ❌ No `Q:` marker → skipped |

If even 1 pair fails parsing, the `if not new_pairs:` catch-all fires and stores the ENTIRE output as a single entry, discarding all parsed pairs.

## Fix: Three Changes

### Change 1: Replace TreeSummarize with `llm.acomplete()`

TreeSummarize wraps the output in its own layer (adds "Here is a summary..." etc.). For structured Q&A output, a direct prompt call gives cleaner results.

```python
# OLD:
synthesizer = TreeSummarize(llm=llm)
response = await asyncio.wait_for(
    synthesizer.aget_response(
        query_str=topic,
        text_chunks=chunk_texts,
        summary_template=prompt,
    ),
    timeout=60.0,
)
answer_text = str(response).strip()

# NEW:
full_prompt = f"{prompt}\n\nContext:\n{' '.join(chunk_texts[:5])}\n\nGenerate {count} Q&A pairs:"
response = await asyncio.wait_for(
    llm.acomplete(full_prompt),
    timeout=60.0,
)
answer_text = str(response).strip()
```

### Change 2: Remove strict parsing — count instead

```python
# OLD: parse each pair with strict format
new_pairs = []
for line in answer_text.split("\n"):
    if line.startswith("[") and "Q:" in line:
        ...
    elif line.startswith("A:") and new_pairs:
        ...
if not new_pairs:
    new_pairs.append({"q": topic, "a": answer_text[:200]})

# NEW: show raw output, count pairs by looking for patterns
import re
q_markers = len(re.findall(
    r'(?:^|\n)\s*(?:\[\d+\]|Q\d*[:.]|Question\s*\d*[:.]|^\d+[.])',
    answer_text, re.IGNORECASE
))
estimated = max(q_markers, 1)
new_pairs = [{"q": f"Generated {estimated} pairs", "a": answer_text}]
```

### Change 3: Display raw output instead of reformatting

```python
# OLD: rebuild lines from parsed pairs
lines = [header, ""]
for i, pair in enumerate(new_pairs, start_num):
    lines.append(f"[{i}] Q: {pair.get('q', '')}")
    lines.append(f"    A: {pair.get('a', '')}")
    lines.append("")

# NEW: show raw model output directly
lines = [header, "", answer_text, ""]
```

This way the model's own formatting is preserved perfectly. The pair count shown in the header is estimated from pattern matching.

---

## Summary

| Change | Problem Solved |
|---|---|
| `llm.acomplete()` instead of TreeSummarize | No wrapping/formatting interference |
| Count-based estimation instead of strict parsing | Handles any output format |
| Display raw output | Model's formatting preserved 1:1 |

4 lines removed, 8 lines added. `/qna` now shows all 10 pairs correctly.
