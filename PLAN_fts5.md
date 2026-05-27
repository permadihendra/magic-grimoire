# Magic Grimoire — PLAN: SQLite FTS5 Analysis

## What FTS5 Is

SQLite's built-in full-text search engine. Provides:
- BM25 relevance ranking (keyword-based, like Google)
- `MATCH` queries with boolean operators (`"world economy" AND takeaway NOT loss`)
- Prefix queries (`world*` matches "worldwide", "worlds")
- Zero config — just `CREATE VIRTUAL TABLE ... USING fts5(...)`

## Where FTS5 Could Help — Ranked by Impact

### 🥇 #1: Hybrid Retrieval (BM25 + Vector) — HIGH IMPACT

**Problem**: Vector search collapses "world economy" and "finance" into similar vectors → wrong document retrieved.

**FTS5 fix**: Use BM25 keyword ranking to BOOST chunks that contain the user's exact words:

```
Vector search returns 20 chunks:
  Finance chunk (0.80) — contains "economy" 0 times
  World Economy chunk (0.40) — contains "world" 2×, "economy" 3×

BM25 re-ranks:
  World Economy chunk: 0.40 + BM25_boost → 0.75 ← RISES
  Finance chunk: 0.80 + 0 BM25_boost → 0.80 ← NO CHANGE

Combined: Finance 0.80 vs World Economy 0.75 → closer but Finance still wins
Need stronger BM25 weight → 0.40 + 0.50 = 0.90 → World Economy WINS!
```

**Implementation**: Build FTS5 index on chunks text. After vector retrieval, run FTS5 MATCH against same chunks, merge scores.

**Cost**: 
- Must extract all chunks into SQLite (currently in LlamaIndex JSON)
- FTS5 index grows with corpus: ~1000 chunks × 512 tokens → ~2 MB
- Latency: FTS5 MATCH over 1000 rows < 1ms

**Verdict**: ✅ Worth it IF we solve the chunk extraction problem. But the simpler approach (keyword pre-filtering we already do in Python) may be sufficient for now.

### 🥈 #2: Document Name Search — MEDIUM IMPACT

**Problem**: User says "world economy book" → our fuzzy matching must find "The_World_Economy_and_Financial_System...". Currently uses word-overlap matching.

**FTS5 fix**: 

```sql
CREATE VIRTUAL TABLE documents_fts USING fts5(
    display_name, filename, 
    content=documents, content_rowid=id
);

-- User query: "world economy"
SELECT display_name, rank FROM documents_fts 
WHERE documents_fts MATCH 'world economy'
ORDER BY rank;
```

BM25 naturally ranks "The World Economy and Financial System" higher than "Finance for Normal People" because it contains the matched words.

**Benefit**: No more fragile regex. Single SQL query. BM25 ranking.

**Verdict**: ✅ Simple to add (just index document names). Replaces ~40 lines of Python fuzzy matching with 1 SQL query.

### 🥉 #3: Knowledge Cache Pre-Filter — LOW IMPACT

**Problem**: 1000+ cached Q&A pairs → cosine over all 1000 takes ~1ms.

**FTS5 fix**: Pre-filter to top-10 keyword matches, then cosine on those 10.

**Verdict**: ❌ Not worth it. 1ms for 1000 entries is already fast. FTS5 adds complexity with zero user-visible benefit.

### ❌ #4: Query History Search — NOT NEEDED

**Use case**: "Show me all queries about world economy."

**Verdict**: ❌ Not a current feature. Add when requested.

---

## Recommended Implementation: #2 Only (Document Name FTS5)

FTS5 on document names gives the best effort-to-impact ratio:

### What It Replaces

```python
# CURRENT: 40 lines of fragile fuzzy matching
def _fuzzy_match_document(user_text: str, filename: str) -> float:
    doc_words = set(user_text.lower().replace("_", " ").split())
    fname_words = set(filename.lower().replace("_", " ").split())
    overlap = len(doc_words & fname_words) / len(doc_words)
    ...
    return overlap

# FTS5: 1 line
SELECT display_name FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT 1
```

### Implementation

```sql
-- In init_db()
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    filename,
    display_name,
    content=documents,
    content_rowid=id
);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER documents_ai AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, filename, display_name) 
    VALUES (new.id, new.filename, new.display_name);
END;

CREATE TRIGGER documents_ad AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, filename, display_name) 
    VALUES ('delete', old.id, old.filename, old.display_name);
END;
```

### Usage in Engine

```python
async def find_document_by_name(name_hint: str) -> str | None:
    """FTS5 search for a document by partial name."""
    db = await get_db()
    # FTS5 query: prefix search for each word
    fts_query = " OR ".join(f"{w}*" for w in name_hint.split() if len(w) > 2)
    
    cursor = await db.execute(
        "SELECT display_name, filename, rank FROM documents_fts "
        "WHERE documents_fts MATCH ? ORDER BY rank LIMIT 3",
        (fts_query,)
    )
    rows = await cursor.fetchall()
    
    if rows:
        # Return best match filename (used as document parameter)
        return rows[0]["filename"]
    return None
```

### Replace Current Matching in Engine

```python
# BEFORE (in query_with_sources and retrieve_only — duplicated logic):
if document:
    doc_words = set(document.lower().replace("_", " ").split())
    stopwords = {...}
    doc_words -= stopwords
    for node in nodes:
        fname = node.metadata.get("file_name", "").lower()
        fname_norm = fname.replace("_", " ").replace(".", " ")
        fname_words = set(fname_norm.split())
        overlap = len(doc_words & fname_words) / len(doc_words)
        ...

# AFTER: Use the exact filename from FTS5 match
if document:
    matched_filename = document  # already matched by FTS5 in BrainPlugin
    for node in nodes:
        fname = node.metadata.get("file_name", "")
        if _filename_match(fname, matched_filename):
            node.score = (node.score or 0) * 2.0  # SIMPLE direct match
```

The document parameter passed by Gemini is first resolved through FTS5 to find the EXACT stored filename. Then the engine just does direct comparison — no fuzzy matching needed.

### What About Chunks FTS5 for Retrieval?

Chunk FTS5 would be powerful but requires:
1. Extracting all chunk texts from LlamaIndex JSON into SQLite
2. Keeping them in sync when index rebuilds
3. Running two searches per query (vector + FTS5) and merging scores

At 1000 chunks, the Python keyword pre-filter we already have is sufficient. FTS5 on chunks becomes worthwhile at 10,000+ chunks — not our scale.

### Files

| File | Change |
|---|---|
| `app/database.py` | Add `documents_fts` virtual table + triggers |
| `app/rag/engine.py` | Simplify doc matching — use exact filename from FTS5 |
| `app/plugins/brain/handler.py` | Use FTS5 to resolve user-specified doc name before passing to ask() |
