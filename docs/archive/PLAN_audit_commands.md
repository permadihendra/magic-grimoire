# PLAN_audit_commands.md — Command UI Audit

## Goal
Audit all bot commands (`/docs`, `/files`, `/index`, `/delete`, `/quiz`) for consistent,
information-rich UI — matching the quality of `/ask` answers which now have:
source citations, next-step suggestions, and retrieval diagnostics.

## Reference: Current /ask answer output (✅ ideal)

```
[Answer text here...]

📖 Primary source: The Bhagavad Gita
📎 Also: Chapter Notes.txt

💡 Next steps: Would you like a follow-up question, a quiz on this topic,
   or a summary of the key points? Just ask!

📊 Retrieval: 3 chunks | ~420 tokens | k=3 | ctx=2048
```

---

## Command-by-Command Audit

### 1. `/docs` (list indexed documents)
**Current output:**
```
📚 Indexed Documents

📄 **The Bhagavad Gita**
   └─ 6.7 MB · 186 chunks · indexed 28 May 2026, 05:08

📄 **Chapter Summary**
   └─ 0.8 MB · 23 chunks · indexed 27 May 2026, 18:00

_Total: 2 documents_
```

**Issues:**
- ❌ No word count per document
- ❌ No parsing method indicator (ebooklib? SimpleDirectoryReader?)
- ❌ No verified/unverified flag (probe status)
- ❌ No display name fallback when `display_name` is NULL in DB
- ❌ Word count column (`word_count`) exists in DB but not queried

**Proposal — `/docs` v2:**
```
📚 *Your Library* (2 documents indexed)

📄 **The Bhagavad Gita**
   └─ 📊 6.7 MB · 186 chunks · 46,677 words · indexed 28 May 2026, 05:08
   └─ ✅ ebooklib (EPUB) · probe verified

📄 **Chapter Summary**
   └─ 📊 1.2 MB · 45 chunks · 8,340 words · indexed 27 May 2026, 18:00
   └─ ⚠️ probe unknown

_Total: 2 documents · ~55,000 words across all materials_
```

**Changes needed:**
1. Query `word_count` column from DB
2. Add parsing method — store in DB as `parse_method` column
3. Add probe/verified status — store in DB as `verified BOOLEAN`
4. Fallback: if `display_name` is NULL, clean filename via `_simple_clean_filename()`
5. Add total summary footer

**DB migration:**
```sql
ALTER TABLE documents ADD COLUMN parse_method TEXT;
ALTER TABLE documents ADD COLUMN verified BOOLEAN DEFAULT 0;
```

---

### 2. `/files` (list all files on disk)
**Current output:**
```
📁 Files in docs directory

[1] `Bhagavad Gita.epub`
    └─ 6.7 MB · ✅ indexed · 28 May 2026, 05:08
[2] `Old Notes.pdf`
    └─ 1.2 MB · ⬜ not indexed · 20 May 2026, 14:00

_Total: 2 files_
_Delete by ID: /delete <number>_
```

**Issues:**
- ❌ No word count for files that ARE indexed
- ❌ No chunk count for indexed files
- ❌ No display name — shows raw filename
- ❌ `display_name` from DB is unused
- ❌ "not indexed" files not distinguished from "failed to parse" files

**Proposal — `/files` v2:**
```
📁 *Files in docs directory* (2 files, 1 indexed)

[1] ✅ `Bhagavad Gita.epub`
     └─ 📊 6.7 MB · 46,677 words · 186 chunks · indexed 28 May 2026
     └─ 💾 The Bhagavad Gita

[2] ⬜ `Old Notes.pdf`
     └─ 📊 1.2 MB · not indexed · 20 May 2026
     └─ ⏳ Not yet parsed (run /index to include)

_Total: 2 files | 1 indexed | 1 pending_
_Delete by ID: /delete <number> | Re-index: /index_
```

**Changes needed:**
1. Join with `documents` table to get `display_name`, `word_count`, `chunk_count`
2. Differentiate "not indexed" vs "indexed with probe OK" vs "probe failed"
3. Show "Run /index" for unindexed files
4. Show display name from DB

---

### 3. `/index` (rebuild index)
**Current output (post-fix):**
```
🔍 Pre-index check: 1 file(s)

📄 [1/1] `Bhagavad Gita.epub` (6.7 MB)
   └─ ebooklib (EPUB) ✅ — 46,677 words

✅ Ready. Building index...

📖 Embedding 847/847 [██████████] 100% — 18s

✅ Index built!
   Docs: 1 | Chunks: 847 | ~61,234 words (18s)
   ✅ Verified

📄 `Bhagavad Gita.epub` — ✅ 46,677 words | 186 chunks

Try `/ask` or `/quiz` to study!
```

**Issues:**
- ⚠️ Minor: "probe verified" footer is at top, not clearly tied to each file
- ⚠️ "⚠️ {failed} file(s) could not be parsed" — when failed > 0, no list of which
- ⚠️ No word count totals across indexed files in summary

**Proposal — `/index` v2 (incremental):**
1. Store `parse_method` in DB per document (already computed in `ParseResult.method`)
2. Store `verified` in DB after probe
3. Add per-file probe status line after chunks:
   ```
   📄 `Bhagavad Gita.epub` — ✅ 46,677 words | 186 chunks | ✅ probe verified
   ```
4. Fix the failed files listing (currently fails to list them)

---

### 4. `/delete` (delete a file)
**Current confirmation:**
```
🗑️ Deleted file #3: `Old Notes.pdf`

Run `/index` to rebuild the index without this file.
```

**Issues:**
- ❌ No word count or chunk count shown before confirming
- ❌ User deletes blind — doesn't know impact
- ❌ No "Are you sure?" — but this is intentional (Telegram already confirms)

**Proposal — `/delete` v2:**
```
🗑️ *Deleted file #3:* `Old Notes.pdf`
   └─ Was: 1.2 MB · 8,340 words · 45 chunks
   └─ Removed from index (run /index to confirm)

Run `/index` to rebuild the index without this file.
```

**Changes needed:**
1. Query DB for word_count and chunk_count before deleting
2. Show impact in confirmation message

---

### 5. `/quiz` (generate practice questions)
**Current output:**
```
📝 *Quiz: Bhagavad Gita — Chapter 1*

**Q1.** [Question text]
A) [Option]  B) [Option]  C) [Option]  D) [Option]
✅ Answer: B

[...more questions...]

📚 Try `/ask` to study specific topics!
```

**Issues:**
- ❌ No source citation (which documents were used?)
- ❌ No retrieval stats (how many chunks used, token budget?)
- ❌ No difficulty indicator shown in output
- ❌ No next-step suggestion after quiz (e.g., "Want a summary?")

**Proposal — `/quiz` v2:**
```
📝 *Quiz: Bhagavad Gita — Chapter 1*

**Q1.** [Question text]
A) [Option]  B) [Option]  C) [Option]  D) [Option]
✅ Answer: B

[...more questions...]

📖 *Sources used:* The Bhagavad Gita (3 chunks)
💡 *Next steps:* Ask a follow-up question, or try `/summarize` for a topic overview.

📊 *Retrieval:* 3 chunks | ~320 tokens | difficulty=normal
```

**Changes needed:**
1. Return source info from `engine.query(mode="quiz")`
2. Add next-step suggestion after quiz output
3. Show retrieval stats footer (consistent with `/ask`)

---

## Cross-cutting Changes

### A. DB Schema Enhancement
```sql
-- Add missing columns
ALTER TABLE documents ADD COLUMN parse_method TEXT;
ALTER TABLE documents ADD COLUMN verified BOOLEAN DEFAULT 0;
```

Update `_sync_documents()` in `indexer.py` to write these:
```python
await db.execute(
    """INSERT OR REPLACE INTO documents
       (filename, filepath, file_size, display_name,
        word_count, chunk_count, parse_method, verified)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
    (fname, fpath, size, display_name,
     word_count, chunk_count, result.method, probe["ok"]),
)
```

### B. `_build_file_card()` helper
Create a shared helper used by both `/docs` and `/files`:

```python
def _build_file_card(row: dict, index: bool = True) -> str:
    """Build a consistent file info card for /docs and /files."""
    name = row.get("display_name") or _simple_clean_filename(row["filename"])
    size_mb = row.get("file_size", 0) / (1024 * 1024)
    chunks = row.get("chunk_count") or 0
    words = row.get("word_count") or 0
    parse_method = row.get("parse_method") or "unknown"
    verified = row.get("verified", False)

    card = [
        f"📄 **{name}**",
        f"   └─ 📊 {size_mb:.1f} MB · {chunks} chunks · {words:,} words",
    ]
    if index:
        status = "✅ probe verified" if verified else "⚠️ probe unknown"
        card.append(f"   └─ {parse_method} · {status}")
    else:
        card.append(f"   └─ {parse_method}")
    return "\n".join(card)
```

---

## Files to Modify

| File | Changes |
|---|---|
| `app/database.py` | Add `parse_method` + `verified` columns to schema |
| `app/rag/indexer.py` | Write `parse_method` + `verified` to DB in `_sync_documents` |
| `app/plugins/study/handler.py` | Fix `_handle_docs` + `_handle_files` + `_handle_delete` + `/quiz` response |
| `app/plugins/brain/handler.py` | Update `_tool_list_docs` with new format |

---

## Priority

1. **P0** — `/docs` + `/files` — most visible, users check these often
2. **P1** — `/delete` confirmation improvement — quick fix
3. **P2** — `/quiz` sources + next-step — consistency with `/ask`

---

## Risk
- Low risk — all changes are UI-only, no logic changes
- Need to handle NULL values in DB (old entries won't have `parse_method`/`verified`)
- Backward compatible: use `.get()` with defaults