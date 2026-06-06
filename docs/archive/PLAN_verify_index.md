# Magic Grimoire — PLAN: Verify Indexing + Fail Loudly

## The Core Problem

The system is **too quiet about failures**. It says "success" when nothing happened.
Users see `/docs` showing books, but `/ask` finds nothing — and they don't know why.

### Failure Modes

| Failure Mode | Current Behavior | Desired Behavior |
|---|---|---|
| EPUB parsing fails | "Indexed 1 files" (lie) | "❌ EPUB parsing failed — try PDF" |
| Index is empty | "0 chunks retrieved" → generic answer | "⚠️ Index empty — no documents parsed" |
| File not parsable | Silent skip + warning | "❌ Could not parse: unsupported format" |
| Index storage corrupt | "Expecting value: line 1 column 1" | "⚠️ Index corrupted — rebuilding" |
| DB entry exists but index empty | `/docs` shows ghost docs | `/docs` shows ⚠️ for unverified docs |

---

## Solution: Verify Before / During / After

### Phase 1: Pre-index — Diagnose Before Starting

```
User: /index

🔍 *Pre-index check...*

📄 Bhagavad Gita.epub (6.7 MB)
  ├─ Format: .epub
  ├─ ebooklib: ✅ installed
  ├─ html2text: ✅ installed
  └─ Parser: epub (ebooklib) → testing...

  ├─ Parse test: ✅ SUCCESS (61,234 words extracted)
  └─ Chunk estimate: ~847 chunks (256 tokens/chunk, 30 overlap)

✅ Ready to index. Starting build...
```

**If parse test fails:**
```
❌ *Pre-index check failed for Bhagavad Gita.epub*

Parsing error: "Fallback parser failed... Please install extra dependencies..."

Suggested fixes (try in order):
1. Run `uv sync --extra epub` to install ebooklib + html2text
2. Convert EPUB to PDF and re-upload
3. Try a different file format (PDF, TXT, MD)

⚠️ Will NOT index — the old index is preserved.
```

### Phase 2: During Index — Real-time Progress

```
📖 *Indexing 1 file...*

[1/1] Bhagavad Gita.epub (6.7 MB)
  ├─ Extracting text... (ebooklib) ✅
  ├─ Words extracted: 61,234
  ├─ Chunking: 847 chunks... (256 tokens, 30 overlap)
  ├─ Embedding: 847/847 [██████████] 100% — 18s
  └─ Saving index...

✅ *Index built successfully!*
   Docs: 1 | Chunks: 847 | Words: ~61K
   Storage: data/index_storage/ (persistable)

⚠️ Note: 0 previously indexed documents were removed.
```

**If some files fail (partial success):**
```
⚠️ *Indexed 1/2 files. 1 file failed.*

✅ Bhagavad Gita.epub: 847 chunks | ~61K words

❌ World Economy.pdf: Parsing failed — "Missing required dependencies"

Suggestions:
  • Run `uv sync --extra liteparse` for better PDF parsing
  • Check that the PDF is not password-protected
  • Try opening the PDF in a browser to verify it's readable
```

### Phase 3: Post-index — Verify It Actually Works

After building the index, run a **probe query** to confirm retrieval works:

```python
def _probe_index(index: VectorStoreIndex) -> dict:
    """Run a test retrieval to verify the index works."""
    try:
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        test_nodes = retriever.retrieve("test query")
        return {
            "ok": len(test_nodes) > 0,
            "chunks_found": len(test_nodes),
            "sample_text": test_nodes[0].text[:50] if test_nodes else None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

If probe fails → tell user the index is broken:
```
❌ *Index verification FAILED*

The index was built but retrieval test failed: "connection refused"

Suggestions:
  • Restart the bot: `./start-bot.sh`
  • Run `/index` again to rebuild
  • Check Ollama is running: `ollama serve`
```

### Phase 4: Fix /docs — Flag Unverified Documents

Current `/docs` shows all documents from DB, even if not indexed.

```
/docs

📚 *Indexed Documents*

📄 **The Bhagavad Gita** (6.7 MB)
   ├─ Chunks: 0 ⚠️ (indexing failed — not in vector store)
   └─ Indexed: 27 May 2025, 21:40

📄 **World Economy** (2.3 MB)
   ├─ Chunks: 847 | ~61K words ✅
   └─ Indexed: 27 May 2025, 21:40

Total: 2 documents | 1 verified | 1 unverified ⚠️

⚠️ 1 document couldn't be verified in the vector index.
   Run `/index` to re-index failed documents.
```

To do this, `/docs` needs to check BOTH the DB and the vector index:

```python
async def _check_doc_in_index(doc_filepath: str) -> int:
    """Check how many chunks exist for this document in the index."""
    try:
        index = await _doc_indexer.ensure_index()
        chunks = [
            n for n in index.docstore.docs.values()
            if n.metadata.get("file_name") == doc_filepath
        ]
        return len(chunks)
    except Exception:
        return -1  # Index corrupt or empty
```

### Phase 5: Fix /files — Flag Unverified

The `/files` command (Telegram file upload) stores files in DB with display_name.
But if `/index` never succeeded, the file exists in DB but not in vector index.

Current `/files` output:
```
📁 *Files in database*

[1] The Bhagavad Gita.pdf — 6.7 MB — indexed 27 May 21:40
[2] World Economy.pdf — 2.3 MB — indexed 27 May 21:40
```

Should add verification:
```
📁 *Files in database*

[1] The Bhagavad Gita.pdf — 6.7 MB — indexed 27 May 21:40
    └─ ⚠️ Not in vector index (0 chunks found) — run /index

[2] World Economy.pdf — 2.3 MB — indexed 27 May 21:40
    └─ ✅ Verified: 847 chunks in vector index

Use /delete <id> to remove unverified files.
```

### Phase 6: Block /index When All Fail

If all files fail to parse:

```python
if all_files_failed:
    return (
        "❌ *Indexing completely failed — 0 documents could be parsed.*\n\n"
        "Failed files:\n"
        "• Bhagavad Gita.epub: 'ebooklib not available'\n\n"
        "Fix options:\n"
        "1. `uv sync --extra epub` — install EPUB support\n"
        "2. `uv sync --extra liteparse` — install PDF/OCR support\n"
        "3. Convert EPUB to PDF and re-upload\n"
        "4. Try with a plain TXT file\n\n"
        "⚠️ Your old index is preserved. Bot is still functional."
    )
```

---

## Alternative Parsing Methods (when primary fails)

When ebooklib fails, suggest alternatives in order:

| Method | Try When | Install |
|---|---|---|
| **ebooklib** (best for EPUB) | Primary — always try first | `uv sync --extra epub` |
| **LiteParse** (PDF + OCR) | ebooklib fails, PDF available | `uv sync --extra liteparse` |
| **PyPDF2** (basic PDF) | LiteParse fails | Built into llama-index-readers-file |
| **Plain text reader** | File is .txt/.md | Built-in |
| **Calibre CLI** (ebook-convert) | All EPUB methods fail | `sudo apt install calibre` |

For each file, try methods in order until one succeeds:

```python
def _try_parse_file(filepath: str, fname: str) -> tuple[bool, str, int]:
    """Try multiple parsing methods, return (success, method, word_count)."""
    methods = [
        ("ebooklib (EPUB)", _try_ebooklib),
        ("LiteParse (PDF)", _try_liteparse),
        ("SimpleDirectoryReader (PDF/TXT)", _try_simple_dir),
        ("Calibre CLI (EPUB→TXT)", _try_calibre),
    ]

    for method_name, parser_fn in methods:
        try:
            result = parser_fn(filepath, fname)
            if result and result.word_count > 100:
                return True, method_name, result.word_count
        except Exception:
            continue

    return False, "none", 0
```

---

## Implementation Summary

| File | Change | Why |
|---|---|---|
| `app/rag/indexer.py` | `_try_parse_file()` with multiple methods | Fallback chain for EPUB |
| `app/rag/indexer.py` | `_probe_index()` verification | Confirm index works after build |
| `app/plugins/study/handler.py` | Pre-index check + per-file progress + block on fail | Fail loudly, not silently |
| `app/plugins/system/handler.py` | `/docs` verifies each doc against vector index | Show ⚠️ for unverified docs |
| `app/plugins/brain/handler.py` | `/files` shows which files are unverified | Clear status on upload state |
| `app/database.py` | Store `word_count` per document | Pre-compute for `/docs` display |

---

## New /index Output (Complete)

```
User: /index

🔍 *Pre-index check: 1 file*

📄 Bhagavad Gita.epub (6.7 MB) — .epub format
  ├─ ebooklib: ✅
  ├─ html2text: ✅
  ├─ Testing parse... ✅ SUCCESS (61,234 words)

✅ Ready. Starting index build...

📖 *Building index: 847 chunks...*

[1/1] Embedding 847/847 [██████████] 100% — 18s

✅ *Index built and verified!*
   Docs: 1 | Chunks: 847 | Words: ~61K

📄 **The Bhagavad Gita**
   └─ ✅ Verified: 847 chunks in vector index
```

---

## Alternative: "Don't Index, Just Tell Me"

When `/index` detects all files are unparseable, offer:

```
❌ *No documents could be parsed.*

Your files:
• Bhagavad Gita.epub — ebooklib not available

Suggestions:
1. Install EPUB support: `uv sync --extra epub`
2. Convert to PDF: use Calibre or online converter
3. Try a plain .txt file instead

Would you like me to download and convert this EPUB to PDF?
Reply YES to try, or upload a different file format.
```

This gives the user agency — they can choose an alternative method.

---

## Decision Needed

**Execute?** This covers:
1. Multi-method parsing fallback (ebooklib → LiteParse → Calibre)
2. Pre-index verify (test parse before building)
3. Per-file progress during index
4. Probe verification after build
5. `/docs` flags unverified documents
6. `/files` flags unindexed files
7. Block on all-fail with suggestions
8. Offer alternative actions when everything fails