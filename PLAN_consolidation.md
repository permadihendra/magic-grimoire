# Magic Grimoire — Pending Plans: Conflict Analysis & Execution Order

## Status: 18 Plans Total

| Status | Plans |
|---|---|
| ✅ **Executed** | improvement_1-6, guardrail, model_optimization, progress_watcher, fix_hang, retrieval_quality (11 plans) |
| 📋 **Pending** | fix_source_citation, clean_filenames, knowledge_cache, fts5 (**4 plans** under review) |
| 🔮 **Future** | streaming_ux, agentic (complex, later) |
| 📄 **Docs** | audit, PLAN.md (reference only) |

---

## Pending Plans — File-Level Conflict Map

```
app/database.py
  ├─ clean_filenames:   ADD COLUMN display_name
  ├─ fts5:              CREATE VIRTUAL TABLE documents_fts + triggers
  └─ knowledge_cache:   CREATE TABLE knowledge_pairs
  → NO CONFLICT: all add different things

app/rag/engine.py — query_with_sources()
  ├─ fix_source_citation:  GROUP-SORT (line 275-285), new labels (line 365-390)
  ├─ fts5:                 SIMPLIFY doc matching (line 240-270), remove fuzzy math
  └─ knowledge_cache:      ADD cache check BEFORE retrieval (line 228)
  → ⚠️ CONFLICT: all three touch the same 50-line block (retrieval + sources)

app/plugins/brain/handler.py
  ├─ clean_filenames:  FTS5-resolve document name before ask()
  ├─ knowledge_cache:  Inject cache context into agent prompt
  └─ (fts5 helper could live here or in engine)
  → ⚠️ LIGHT CONFLICT: both modify agent prompt + pre-tool logic

app/plugins/study/handler.py
  ├─ clean_filenames:  /files shows display_name
  └─ knowledge_cache:  Invalidate cache on /index
  → NO CONFLICT: different functions

app/rag/knowledge_cache.py (NEW)
  → Standalone — no conflicts
```

---

## Exact Conflict: `query_with_sources()` in engine.py

All three pending plans want to modify this function:

```
LINE 228:  def query_with_sources(self, question, difficulty, document, chat_id):
LINE 230:      retriever = self._get_retriever()
LINE 231:      nodes = retriever.retrieve(question)
          
              ┌─ KNOWLEDGE_CACHE: insert cache check HERE (before retrieval)
              
LINE 234:      # Load feedback learner
LINE 240:      # Document-aware boosting
              
              ┌─ FTS5: replace fuzzy doc_words math with simple exact match
              │  (document already resolved to exact filename by FTS5)
              
LINE 250:      doc_words = set(document.lower()...)  ← FTS5 REMOVES THIS
LINE 260:      for node in nodes: overlap = ...       ← FTS5 SIMPLIFIES THIS
              
LINE 265:      # Feedback adjustments
LINE 275:      # Post-retrieval filtering → doc_nodes + other_nodes
LINE 285:      nodes.sort(key=lambda n: n.score)      ← FIX_CITATION CHANGES THIS
              
LINE 290:      # Extract sources from nodes
LINE 295:      for node in nodes: sources.append(...)
              
LINE 320:      # LLM generation
LINE 365:      # Build source citations
LINE 370:      for s in cited: citations.append(f"📖 Source: {short}")  ← FIX_CITATION CHANGES THIS
```

### Resolution: Correct Execution Order

```
Step 1: FTS5 FIRST
  → Simplifies doc matching to one exact comparison
  → Removes doc_words, fname_words, overlap math
  → Makes subsequent changes easier (less code to modify)

Step 2: fix_source_citation SECOND  
  → Group-sort on the already-simplified code
  → New labels use display_name (but fallback to shorten_filename)
  → Fewer lines to change because FTS5 already cleaned up

Step 3: knowledge_cache THIRD
  → Cache check inserted before retrieval
  → Independent of how retrieval works internally
  → If retrieval changes (steps 1-2), cache layer doesn't care

Step 4: clean_filenames FOURTH
  → Add display_name to DB (can be done early — no conflict)
  → display_name used by FTS5 (step 1) and source labels (step 2)
```

---

## Merged Implementation Order

| Step | Plan | Files | Depends On |
|---|---|---|---|
| **1** | clean_filenames (DB only) | `database.py` | Nothing — safe standalone |
| **2** | fts5 | `database.py`, `engine.py`, `brain/handler.py` | Step 1 (display_name exists) |
| **3** | fix_source_citation | `engine.py` | Step 2 (simplified matching in place) |
| **4** | knowledge_cache | `knowledge_cache.py`, `database.py`, `engine.py`, `brain/handler.py` | Steps 1-3 (stable retrieval) |

### Why This Order

```
clean_filenames DB  →  FTS5 (searches display_name)  →  fix citations (uses display_name + simple matching)
                                                           ↓
                                                    knowledge_cache (sits BEFORE all of it)
```

Knowledge cache goes last because:
- It's a NEW layer, independent of retrieval internals
- If retrieval breaks during steps 1-3, cache still works (it's read-only)
- If cache has bugs, retrieval still works (falls through)

---

## Non-Conflicting Parallel Work

These can be done simultaneously if needed:

```
Thread A: database.py changes (clean_filenames + fts5 + knowledge_cache tables)
Thread B: engine.py changes (must be sequential — FTS5 → citation → cache)
Thread C: brain/handler.py changes (agent prompt updates, cache injection)
```

---

## Summary

| Risk | Mitigation |
|---|---|
| 3 plans touch `query_with_sources()` | Strict order: FTS5 → citation → cache |
| DB schema evolves 3 times | All additive (ALTER TABLE, CREATE TABLE) — no data migration |
| Gemini agent prompt changes | Only append (cache context, FTS5 rules) — no removal |
| display_name depends on Gemini | Fallback to raw filename if Gemini fails |
