# Magic Grimoire — Source Citation: Complete Audit + Fix Plan

## All Citation Paths Traced

### Path 1: `query_with_sources()` → `_tool_ask()` → slow-path → Telegram

```
engine.query_with_sources()
  ├─ retrieve nodes
  ├─ doc boosting
  ├─ feedback adjustments
  ├─ post-retrieval filtering → doc_nodes FIRST
  ├─ ❌ BUG: sort by score UNDOES doc_nodes priority
  ├─ extract sources from sorted nodes → first source may be WRONG doc
  ├─ generate answer
  ├─ build citation block from sources list
  └─ return {answer, sources}

_tool_ask()
  └─ returns answer string (with citations embedded)

slow-path
  ├─ result_lines.append(tool result)
  ├─ final_reply = "\n\n".join(result_lines)
  └─ ❌ BUG: if 2 ask() calls, merges TWO separate citation blocks
```

### Path 2: `retrieve_only()` → `_tool_retrieve()` → preview display

```
engine.retrieve_only()
  ├─ retrieve nodes
  ├─ doc boosting
  ├─ feedback adjustments  
  ├─ post-retrieval filtering → doc_nodes FIRST
  ├─ ❌ BUG: sort by score UNDOES doc_nodes priority (line 150)
  └─ return passages list (doc order now WRONG)

slow-path preview
  └─ shows first 3 passages → may show WRONG doc first
```

### Path 3: `query()` (legacy, no sources) → quiz/summarize

```
engine.query()
  └─ ❌ GAP: no source citation at all. User doesn't know which doc was used.
```

---

## All Issues Found

| # | Severity | Location | Issue |
|---|---|---|---|
| 1 | 🔴 Critical | `query_with_sources()` line 285 | Sort by score undoes post-filtering → wrong source cited first |
| 2 | 🔴 Critical | `retrieve_only()` line 150 | Same bug — sort undoes filtering |
| 3 | 🟡 Medium | `query_with_sources()` Phase 3 | Citation label: "📖 Source: X" doesn't indicate primary vs supplementary |
| 4 | 🟡 Medium | `_tool_ask()` + slow-path | If 2 ask() calls, two separate citation blocks concatenated → "message too long" + confusing |
| 5 | 🟠 Minor | `shorten_filename()` | May over-truncate compound names like "World Economy and Financial System" → "World Economy" → ambiguous with "World Economy And Finance" |
| 6 | 🟢 Gap | `query()` (quiz/summarize) | No source citation at all |
| 7 | 🟢 Gap | `sources` in error paths | `OllamaBusyError` returns `sources` list that may be stale/empty |

---

## Fix Plan

### Fix 1: Group-Sort (Critical — both methods)

Replace global sort with group-sort after filtering:

```python
# BEFORE (bug — sort undoes filtering):
if document and doc_nodes:
    nodes = doc_nodes[:2] + other_nodes[:3]
nodes.sort(key=lambda n: n.score or 0, reverse=True)

# AFTER (group-sort — preserve priority):
if document and doc_nodes:
    doc_nodes.sort(key=lambda n: n.score or 0, reverse=True)
    other_nodes.sort(key=lambda n: n.score or 0, reverse=True)
    nodes = doc_nodes[:2] + other_nodes[:3]
else:
    nodes.sort(key=lambda n: n.score or 0, reverse=True)
```

Apply in BOTH: `query_with_sources()` and `retrieve_only()`.

### Fix 2: Meaningful Source Labels

Instead of just "📖 Source: X", distinguish primary from supplementary:

```python
if document and sources:
    # First source is the named doc (guaranteed by group-sort)
    primary = shorten_filename(sources[0]["filename"])
    citations = [f"📖 *Primary source:* {primary}"]
    
    # Remaining sources are supplementary
    for s in sources[1:]:
        short = shorten_filename(s["filename"])
        score = s["score"]
        if score < 0.7:
            citations.append(f"📎 Also: {short} ⚠️ low relevance")
        else:
            citations.append(f"📎 Also: {short}")
else:
    # No document filter — just list sources normally
    for s in sources:
        short = shorten_filename(s["filename"])
        if s["score"] < 0.7:
            citations.append(f"📖 *Source:* {short} ⚠️ low relevance")
        else:
            citations.append(f"📖 *Source:* {short}")
```

**Result:**
```
Before:  📖 Source: Finance for Normal People
         📖 Source: World Economy

After:   📖 Primary source: World Economy (0.92)
         📎 Also: Finance for Normal People ⚠️ low relevance
```

### Fix 3: Consolidate Multiple Tool Results

When the slow-path executes multiple tools (2 ask() calls), merge their source citations:

```python
# In slow-path after tool execution:
if len(result_lines) > 1:
    # Multiple tool results — consolidate: answer + single citation block
    answer_text = "\n\n".join(result_lines)
    # Strip duplicate/conflicting citations, pick best sources
    ...
```

But better: Fix 1 (agent prompt) already prevents multiple ask() calls. This is defense-in-depth.

### Fix 4: `shorten_filename()` Safety

Add a fallback: if shortened name is < 3 chars, use the raw filename (truncated):

```python
def shorten_filename(filename: str) -> str:
    result = _shorten(filename)  # existing logic
    if len(result) < 3:
        result = filename[:40]   # fallback
    return result
```

### Fix 5: Quiz/Summarize Sources (Optional)

Add source extraction to `query()` and return with citations. Low priority — quiz/summarize are less common.

### Fix 6: Stale Sources in Error Paths

When returning error, set sources to empty or mark them clearly:

```python
except OllamaBusyError as e:
    return {"answer": str(e), "sources": []}  # was: sources (stale)
```

---

## Files

| File | Change |
|---|---|
| `app/rag/engine.py` | Fix 1 (group-sort) in `query_with_sources()` and `retrieve_only()` |
| `app/rag/engine.py` | Fix 2 (meaningful labels) in citation builder |
| `app/rag/engine.py` | Fix 4 (`shorten_filename()` safety) |
| `app/rag/engine.py` | Fix 6 (stale sources in error returns) |
