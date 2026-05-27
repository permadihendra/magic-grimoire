# Magic Grimoire — Improvement Plan #1

## 1. Delete Files by ID (instead of name)

### Problem
`/delete Practical Laravel` requires typing the filename. Case-sensitive, easy to typo, partial match can be ambiguous.

### Solution
Assign numeric IDs to files in `/files` output. Delete by ID.

**Before:**
```
/files
📄 Finance for Normal People.pdf   3.4 MB · indexed
📄 Practical Laravel.pdf           7.2 MB · not indexed

/delete Practical Laravel
⚠️ Multiple files match...
```

**After:**
```
/files
[1] 📄 Finance for Normal People.pdf   3.4 MB · indexed
[2] 📄 Practical Laravel.pdf           7.2 MB · not indexed

/delete 2
🗑️ Deleted file #2: Practical Laravel.pdf
```

### Implementation
- `app/plugins/study/handler.py` — `_handle_files()` adds `[N]` prefix
- `app/plugins/study/handler.py` — `_handle_delete()` accepts numeric ID

---

## 2. Fix Polarized Response (Gemini + RAG Conflict)

### Problem
Agent prompt tells Gemini to "respond conversationally first, then add tool at END." Gemini writes its own answer from training data BEFORE the RAG tool runs. If RAG result differs, user sees conflicting/polarized response:

```
User: "what is loss aversion?"

Gemini: "Great question! Loss aversion is a cognitive bias..."
     → TOOL: ask(query="loss aversion")

User sees: [Gemini's generic answer] + [RAG passages — maybe unrelated]
```

### Solution
Strict separation: Gemini does NOT write the answer for `ask()`/`quiz()`/`summarize()` tools. The RAG tool output IS the answer. Gemini only adds lightweight metadata after the tool.

**After:**
```
User: "what is loss aversion?"

Gemini: TOOL: ask(query="loss aversion")
     → (no conversational text before)

RAG result: 📖 From Finance for Normal People
            Loss aversion is the tendency...

Gemini: 💡 Want me to quiz you on this?
```

### Implementation
- `app/plugins/brain/handler.py` — Update agent prompt:
  - Remove "respond conversationally first" instruction for `ask()`/`quiz()`/`summarize()`
  - Add: "For TOOL: ask/quiz/summarize — do NOT add any text before the tool. The tool output IS the answer."
  - Allow metadata text AFTER the tool (suggestions, follow-ups)

---

## 3. Low-Confidence Warning

### Problem
RAG engine returns chunks even when they're poorly matched. If the top similarity score is low (< 0.7), the answer is unreliable but presented authoritatively.

### Solution
Add a confidence check in the RAG engine. If top source score < 0.7, append a warning.

**Before:**
```
Loss aversion is the tendency to prefer avoiding losses...
```

**After:**
```
Loss aversion is the tendency to prefer avoiding losses...

⚠️ Low confidence — your documents may not cover this topic well.
```

### Implementation
- `app/rag/engine.py` — In `query_with_sources()`, check if any source has score < 0.7. If all sources are below threshold, append warning.

---

## 4. EPUB & E-Book Format Support

### Problem
Currently only PDF and text-based formats work. No EPUB, MOBI, or other e-book formats.

### Solution
Add `.epub` to supported extensions. Both LiteParse and SimpleDirectoryReader can handle EPUB:

| Parser | EPUB | Other e-book |
|---|---|---|
| LiteParse | ✅ Via LibreOffice | ✅ MOBI, AZW3 via LibreOffice |
| SimpleDirectoryReader | ✅ Via `EpubReader` | Limited |

### Implementation
- `app/bot/dispatcher.py` — Add `.epub` to `allowed_extensions` in `_handle_document()`
- `app/rag/indexer.py` — EPUB is handled automatically by `SimpleDirectoryReader` (falls through to `EpubReader`)
- `pyproject.toml` — Ensure `llama-index-readers-file` is available for EPUB support

---

## 5. Upgrade Parser: LiteParse + PyMuPDF Fallback

### Motivation
GPU machine with no resource limits. Current `SimpleDirectoryReader` (PyMuPDF) is basic — no OCR, no spatial text, no bytes input. LiteParse is the best quality-to-effort parser for this environment.

### LiteParse vs Others

| Parser | Quality | Speed | Local | Best For |
|---|---|---|---|---|
| **LiteParse** ★ | 🥇 High | Fast | ✅ Yes | **Default — spatial + OCR + bytes + EPUB** |
| PyMuPDF / SimpleDirectoryReader | 🥉 Basic | Fastest | ✅ Yes | Fallback (also handles EPUB via EpubReader) |
| Marker | 🥇 Highest | Slow | ✅ Yes | Overkill for study docs |
| LlamaParse | 🥇 Highest | Fast | ❌ Cloud | Expensive at scale |

### What LiteParse Adds

| Feature | Before (PyMuPDF) | After (LiteParse) |
|---|---|---|
| **Spatial text** | Sequential dump | Headings, body, footnotes preserved → **better chunks** |
| **OCR** | ❌ Scanned PDFs return empty | ✅ Tesseract OCR reads them |
| **Bytes input** | File path only | `parse(pdf_bytes)` — Telegram upload → parse in one step |

### Flow

```
User uploads PDF → Telegram → bytes
                          ↓
        try: LiteParse(ocr_enabled=True).parse(bytes)
        except ImportError: SimpleDirectoryReader(input_files=[path])
                          ↓
                  LlamaIndex Document objects
                          ↓
                    Chunk → Embed → Index
```

### Implementation

**`app/rag/indexer.py`** — Replace `SimpleDirectoryReader` in `_build_index()`:

```python
from llama_index.core import Document as LIDocument

documents = []
for fpath in file_paths:
    with open(fpath, "rb") as f:
        raw_bytes = f.read()
    doc = _parse_document(raw_bytes, fpath)
    if doc:
        documents.append(doc)

def _parse_document(raw_bytes: bytes, fpath: str) -> LIDocument | None:
    fname = os.path.basename(fpath)
    # Try LiteParse first
    try:
        from liteparse import LiteParse
        parser = LiteParse(ocr_enabled=True)
        result = parser.parse(raw_bytes)
        text = result.text.strip()
        if text:
            return LIDocument(text=text, metadata={"file_name": fname})
    except ImportError:
        pass
    except Exception as e:
        logger.warning("LiteParse failed for %s: %s", fname, e)

    # Fallback: SimpleDirectoryReader (reads from file path)
    try:
        reader = SimpleDirectoryReader(input_files=[fpath])
        docs = reader.load_data()
        if docs:
            return docs[0]
    except Exception as e:
        logger.warning("Fallback parser also failed for %s: %s", fname, e)

    return None
```

### Dependencies
```toml
# pyproject.toml — optional extras
liteparse = {version = "*", optional = true}
[project.optional-dependencies]
liteparse = ["liteparse"]
```

### Why Not Marker or LlamaParse
- **Marker**: GPU-based, excellent quality but 10× slower. For study books with standard layouts, LiteParse gives 95% quality at 10% the time.
- **LlamaParse**: Cloud API — documents leave your machine. Expensive at scale. Free tier rate-limited. Unnecessary when you have a local GPU.

---

## Implementation Order

| Priority | Fix | File(s) | Effort |
|---|---|---|---|
| 🥇 | Delete by ID | `study/handler.py` | ~20 min |
| 🥇 | Fix polarized response | `brain/handler.py` (prompt) | ~10 min |
| 🥈 | Low-confidence warning | `rag/engine.py` | ~10 min |
| 🥈 | EPUB support | `dispatcher.py` (extensions) | ~5 min |
| 🥉 | LiteParse integration | `rag/indexer.py` + pyproject.toml | ~30 min |
