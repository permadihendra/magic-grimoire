"""Document indexer — ingest, chunk, embed, and persist documents.

Features:
- Multi-method parsing: ebooklib → LiteParse → Calibre CLI → SimpleDirectoryReader
- Pre-index verify: test-parse each file before building
- Per-file progress reporting
- Post-build probe verification
- Word count tracking per document
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

from llama_index.core import Document as LIDocument
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from app.config import settings
from app.database import get_db
from app.rag.models import configure_settings

logger = logging.getLogger(__name__)

INDEX_STORAGE_DIR = os.path.join(os.path.dirname(settings.db_path), "index_storage")

# ── Dataclass: parse result ───────────────────────────────

@dataclass
class ParseResult:
    """Result of parsing a document."""
    success: bool
    method: str
    word_count: int
    char_count: int
    error: str = ""
    doc: LIDocument | None = None

# ── Filename Cleaner ─────────────────────────────────────

_FILENAME_CLEAN_PROMPT = """Extract a clean, readable book title from this filename.
Remove: author names, publisher info, years in parentheses, edition numbers, file extensions.
Return ONLY the title, nothing else. No quotes, no prefixes.

Examples:
"Statman, Meir - Finance for normal people..." -> Finance for Normal People
"The_World_Economy_and_Financial_System_A_Paradigm_Change_Offering.epub"
-> The World Economy and Financial System: A Paradigm Change Offering

Now clean this filename:"""


async def _gemini_clean_filename(raw_filename: str) -> str:
    """Ask Gemini to clean a filename into a book title."""
    try:
        from app.llm.gemini import gemini_chat
        response = await gemini_chat(
            system_prompt=_FILENAME_CLEAN_PROMPT,
            user_message=raw_filename,
            max_tokens=50,
            timeout=8.0,
        )
        cleaned = response.strip().strip('"').strip("'")
        if cleaned and len(cleaned) >= 3:
            logger.info("Gemini cleaned filename: %s -> %s", raw_filename[:50], cleaned)
            return cleaned
    except Exception as e:
        logger.debug("Gemini filename clean failed: %s", e)
    return _simple_clean_filename(raw_filename)


def _simple_clean_filename(raw: str) -> str:
    """Simple filename cleaning without Gemini."""
    name = raw.rsplit(".", 1)[0]
    for sep in ["_", "-", "\u2013", "\u2014"]:
        name = name.replace(sep, " ")
    name = " ".join(name.split())
    if len(name) > 50:
        name = name[:47].rsplit(" ", 1)[0] + "..."
    return name.strip()

# ── Multi-method Parsers ─────────────────────────────────

def _try_ebooklib(raw_bytes: bytes, fname: str) -> ParseResult:
    """Try parsing with ebooklib (best for EPUB)."""
    try:
        from ebooklib import epub
        import html2text

        # Parse EPUB
        book = epub.read_epub(fname if os.path.exists(fname) else None,
                              {"epub": raw_bytes})

        text_parts = []
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True

        for item in book.get_items():
            if item.get_type() == 9:  # EPUB_HTML
                html = item.get_content().decode("utf-8", errors="ignore")
                text_parts.append(h.handle(html))

        text = "\n\n".join(t.strip() for t in text_parts if t.strip())
        if text and len(text) > 100:
            word_count = len(text.split())
            doc = LIDocument(text=text, metadata={"file_name": os.path.basename(fname)})
            logger.info("ebooklib parsed %s: %d words", fname, word_count)
            return ParseResult(success=True, method="ebooklib (EPUB)",
                               word_count=word_count, char_count=len(text), doc=doc)
        return ParseResult(success=False, method="ebooklib (EPUB)",
                           word_count=0, char_count=0, error="empty text")
    except ImportError:
        return ParseResult(success=False, method="ebooklib (EPUB)",
                           word_count=0, char_count=0, error="ebooklib not installed")
    except Exception as e:
        return ParseResult(success=False, method="ebooklib (EPUB)",
                           word_count=0, char_count=0, error=str(e))


def _try_liteparse(raw_bytes: bytes, fname: str) -> ParseResult:
    """Try parsing with LiteParse (PDF + OCR)."""
    try:
        from liteparse import LiteParse

        parser = LiteParse(ocr_enabled=True, quiet=True)
        result = parser.parse(raw_bytes)
        text = result.text.strip()
        if text and len(text) > 100:
            word_count = len(text.split())
            doc = LIDocument(text=text, metadata={"file_name": os.path.basename(fname)})
            logger.info("LiteParse parsed %s: %d words", fname, word_count)
            return ParseResult(success=True, method="LiteParse (PDF/OCR)",
                               word_count=word_count, char_count=len(text), doc=doc)
        return ParseResult(success=False, method="LiteParse (PDF/OCR)",
                           word_count=0, char_count=0, error="empty text")
    except ImportError:
        return ParseResult(success=False, method="LiteParse (PDF/OCR)",
                           word_count=0, char_count=0, error="LiteParse not installed")
    except Exception as e:
        return ParseResult(success=False, method="LiteParse (PDF/OCR)",
                           word_count=0, char_count=0, error=str(e))


def _try_simple_dir(fname: str) -> ParseResult:
    """Try parsing with SimpleDirectoryReader (PDF, TXT, MD, etc.)."""
    try:
        reader = SimpleDirectoryReader(input_files=[fname])
        docs = reader.load_data()
        if docs and docs[0].text and docs[0].text.strip():
            for d in docs:
                # Update metadata with original filename (not temp path)
                pass
            text = docs[0].text
            word_count = len(text.split())
            doc = LIDocument(text=text, metadata={"file_name": os.path.basename(fname)})
            logger.info("SimpleDirectoryReader parsed %s: %d words", fname, word_count)
            return ParseResult(success=True, method="SimpleDirectoryReader (PDF/TXT)",
                               word_count=word_count, char_count=len(text), doc=doc)
        return ParseResult(success=False, method="SimpleDirectoryReader (PDF/TXT)",
                           word_count=0, char_count=0, error="empty text")
    except Exception as e:
        return ParseResult(success=False, method="SimpleDirectoryReader (PDF/TXT)",
                           word_count=0, char_count=0, error=str(e))


def _try_calibre(fname: str) -> ParseResult:
    """Try parsing with Calibre CLI (ebook-convert → TXT)."""
    try:
        # Check if calibre is available
        result = subprocess.run(
            ["which", "ebook-convert"], capture_output=True, timeout=5
        )
        if result.returncode != 0:
            return ParseResult(success=False, method="Calibre CLI (EPUB→TXT)",
                               word_count=0, char_count=0, error="calibre not installed")
    except Exception:
        return ParseResult(success=False, method="Calibre CLI (EPUB→TXT)",
                           word_count=0, char_count=0, error="calibre not available")

    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        # Convert EPUB to TXT using Calibre
        r = subprocess.run(
            ["ebook-convert", fname, tmp_path],
            capture_output=True, text=True, timeout=60
        )
        if r.returncode != 0:
            os.unlink(tmp_path)
            return ParseResult(success=False, method="Calibre CLI (EPUB→TXT)",
                               word_count=0, char_count=0, error=r.stderr[:200])

        with open(tmp_path, "r", errors="ignore") as f:
            text = f.read().strip()

        os.unlink(tmp_path)

        if text and len(text) > 100:
            word_count = len(text.split())
            doc = LIDocument(text=text, metadata={"file_name": os.path.basename(fname)})
            logger.info("Calibre parsed %s: %d words", fname, word_count)
            return ParseResult(success=True, method="Calibre CLI (EPUB→TXT)",
                               word_count=word_count, char_count=len(text), doc=doc)
        return ParseResult(success=False, method="Calibre CLI (EPUB→TXT)",
                           word_count=0, char_count=0, error="empty output")
    except subprocess.TimeoutExpired:
        return ParseResult(success=False, method="Calibre CLI (EPUB→TXT)",
                           word_count=0, char_count=0, error="timeout (60s)")
    except Exception as e:
        return ParseResult(success=False, method="Calibre CLI (EPUB→TXT)",
                           word_count=0, char_count=0, error=str(e))


def parse_document_multimethod(
    fpath: str, fname: str, raw_bytes: bytes | None = None
) -> ParseResult:
    """Try multiple parsing methods in order until one succeeds.

    Methods tried (in order):
    1. ebooklib — best for EPUB
    2. LiteParse — PDF + OCR
    3. Calibre CLI — EPUB via conversion
    4. SimpleDirectoryReader — PDF, TXT, MD, etc.

    Returns ParseResult with success=True and doc, or success=False with error.
    """
    ext = os.path.splitext(fname)[1].lower()

    # Method priority based on file type
    methods_by_ext = {
        ".epub": [_try_ebooklib, _try_liteparse, _try_calibre, _try_simple_dir],
        ".pdf": [_try_liteparse, _try_simple_dir],
        ".txt": [_try_simple_dir],
        ".md": [_try_simple_dir],
        ".docx": [_try_simple_dir],
        ".doc": [_try_simple_dir],
    }

    methods = methods_by_ext.get(ext, [_try_simple_dir])

    # For methods that need a file path, write bytes to temp
    need_file = {_try_simple_dir, _try_calibre}

    for method_fn in methods:
        if method_fn in need_file:
            if raw_bytes is None:
                try:
                    with open(fpath, "rb") as f:
                        raw_bytes = f.read()
                except Exception:
                    pass

            if raw_bytes is not None and method_fn in {_try_ebooklib}:
                # ebooklib needs file path
                result = method_fn(raw_bytes, fpath)
            elif method_fn == _try_simple_dir:
                result = method_fn(fpath)
            else:
                result = method_fn(raw_bytes, fpath) if raw_bytes else ParseResult(
                    success=False, method="unknown", word_count=0, char_count=0,
                    error="no file bytes"
                )
        else:
            if raw_bytes is None:
                try:
                    with open(fpath, "rb") as f:
                        raw_bytes = f.read()
                except Exception:
                    pass
            result = method_fn(raw_bytes, fpath) if raw_bytes else ParseResult(
                success=False, method="unknown", word_count=0, char_count=0,
                error="no file bytes"
            )

        if result.success:
            return result

    # All methods failed
    return ParseResult(success=False, method="none", word_count=0, char_count=0,
                       error="all parsing methods failed")

# ── Index Verification ────────────────────────────────────

def probe_index(index: VectorStoreIndex) -> dict:
    """Run a test retrieval to verify the index is actually usable."""
    try:
        from llama_index.core.retrievers import VectorIndexRetriever
        retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
        test_nodes = retriever.retrieve("test query")
        sample_text = ""
        if test_nodes and test_nodes[0].text:
            sample_text = test_nodes[0].text[:80]
        return {
            "ok": True,
            "chunks_found": len(test_nodes),
            "sample_text": sample_text,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── DocumentIndexer ───────────────────────────────────────

class DocumentIndexer:
    """Handles document ingestion and index management."""

    def __init__(self) -> None:
        self._index: VectorStoreIndex | None = None
        self._parse_results: list[ParseResult] = []
        self._parse_errors: list[tuple[str, str]] = []
        self._last_build_stats: dict | None = None

    @property
    def index(self) -> VectorStoreIndex | None:
        return self._index

    @property
    def last_build_stats(self) -> dict | None:
        return self._last_build_stats

    @property
    def parse_results(self) -> list[ParseResult]:
        return self._parse_results

    @property
    def parse_errors(self) -> list[tuple[str, str]]:
        return self._parse_errors

    async def ensure_index(self) -> VectorStoreIndex:
        """Load existing index or build a new one if none exists."""
        if self._index is not None:
            return self._index

        configure_settings()

        if os.path.exists(INDEX_STORAGE_DIR):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=INDEX_STORAGE_DIR
                )
                self._index = load_index_from_storage(storage_context)
                logger.info("Loaded existing index")
                return self._index
            except Exception as e:
                logger.warning("Failed to load existing index: %s", e)

        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)

        self._index = await self._build_index(docs_dir, report_fn=None)
        return self._index

    async def rebuild_index(self, report_fn=None) -> VectorStoreIndex:
        """Force rebuild the index. report_fn(chunk_count) is called for progress."""
        configure_settings()

        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)

        if os.path.exists(INDEX_STORAGE_DIR):
            shutil.rmtree(INDEX_STORAGE_DIR)
            logger.info("Removed old index storage")

        self._index = await self._build_index(docs_dir, report_fn=report_fn)
        return self._index

    # ── Pre-index check: test-parse each file ──────────────

    async def pre_index_check(self, docs_dir: str) -> list[tuple[str, ParseResult]]:
        """Test-parse each file before building the index.

        Returns list of (filename, ParseResult) for all files.
        """
        SUPPORTED_EXTS = {
            ".pdf", ".txt", ".md", ".docx", ".doc",
            ".epub", ".rtf", ".csv", ".json", ".xml",
            ".html", ".htm", ".pptx", ".ppt", ".xlsx", ".xls",
        }

        files = [
            f for f in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, f))
            and not f.startswith(".")
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        ]

        results = []
        for fname in files:
            fpath = os.path.join(docs_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    raw_bytes = f.read()
                result = parse_document_multimethod(fpath, fname, raw_bytes)
                results.append((fname, result))
            except Exception as e:
                results.append((fname, ParseResult(
                    success=False, method="none", word_count=0, char_count=0,
                    error=f"read error: {e}"
                )))

        return results


    async def pre_index_check_single(self, fpath: str, fname: str) -> ParseResult:
        """Test-parse a single file to check if it's parsable."""
        try:
            with open(fpath, "rb") as f:
                raw_bytes = f.read()
            return parse_document_multimethod(fpath, fname, raw_bytes)
        except Exception as e:
            return ParseResult(success=False, method="none", word_count=0, char_count=0,
                               error=f"read error: {e}")

    async def _build_index(self, docs_dir: str, report_fn=None) -> VectorStoreIndex:
        """Build a new index from documents in the given directory."""
        SUPPORTED_EXTS = {
            ".pdf", ".txt", ".md", ".docx", ".doc",
            ".epub", ".rtf", ".csv", ".json", ".xml",
            ".html", ".htm", ".pptx", ".ppt", ".xlsx", ".xls",
        }

        files = [
            f for f in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, f))
            and not f.startswith(".")
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        ]

        self._parse_results = []
        self._parse_errors = []

        if not files:
            logger.warning("No documents found in %s", docs_dir)
            dummy = Document(text="")
            index = VectorStoreIndex.from_documents([dummy])
            index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
            self._index = index
            self._last_build_stats = {"docs": 0, "chunks": 0, "words": 0, "failed": 0}
            return index

        start = time.time()

        # ── Parse each file ────────────────────────────────
        documents = []
        for fname in files:
            fpath = os.path.join(docs_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    raw_bytes = f.read()
                result = parse_document_multimethod(fpath, fname, raw_bytes)
                self._parse_results.append(result)

                if result.success and result.doc:
                    documents.append(result.doc)
                else:
                    self._parse_errors.append((fname, result.error or "unknown"))
                    logger.warning("Skipping unparseable file: %s — %s", fname, result.error)
            except Exception as e:
                self._parse_errors.append((fname, str(e)))
                logger.warning("Failed to read %s: %s", fname, e)

        if not documents:
            logger.error("No documents could be parsed!")
            dummy = Document(text="")
            index = VectorStoreIndex.from_documents([dummy])
            index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
            self._index = index
            self._last_build_stats = {
                "docs": 0, "chunks": 0, "words": 0, "failed": len(files)
            }
            return index

        logger.info("Parsed %d/%d documents successfully", len(documents), len(files))

        # ── Embed documents in batches ─────────────────────
        BATCH_SIZE = 50

        from app.rag.guard import OllamaGuard

        try:
            async with OllamaGuard("document indexing", timeout=600):
                if len(documents) <= BATCH_SIZE:
                    index = VectorStoreIndex.from_documents(documents, show_progress=True)
                else:
                    all_nodes = []
                    for i in range(0, len(documents), BATCH_SIZE):
                        batch = documents[i:i+BATCH_SIZE]
                        batch_index = VectorStoreIndex.from_documents(
                            batch, show_progress=True
                        )
                        all_nodes.extend(batch_index.docstore.docs.values())
                        chunk_count = len(all_nodes)
                        if report_fn:
                            report_fn(chunk_count)

                    index = VectorStoreIndex(
                        nodes=all_nodes,
                        embed_model=batch_index._embed_model if all_nodes else None,
                    )

            index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)

        except Exception as e:
            logger.error("Indexing failed: %s", e)
            # Detect Ollama embed crash (Go panic = ResponseError with "signal arrived")
            from ollama._types import ResponseError
            if isinstance(e, ResponseError) and "signal" in str(e).lower():
                logger.critical(
                    "Ollama embed Go runner CRASHED during indexing. "
                    "This is a known issue when Ollama is under GPU+CPU load during batch embedding. "
                    "Recommend: use sentence-transformers fallback (already configured). "
                    "If this persists, reduce chunk_size or switch to CPU-only embedding."
                )
                raise RuntimeError(
                    f"Ollama embed crashed during indexing: {e}\n\n"
                    "The fallback embedding (sentence-transformers all-MiniLM-L6-v2) should take over. "
                    "If this message persists after restart, run: uv add sentence-transformers"
                ) from e
            raise RuntimeError(f"Indexing failed: {e}") from e

        elapsed = time.time() - start

        # ── Post-build probe verification ──────────────────
        probe = probe_index(index)
        if not probe["ok"]:
            logger.error("Index probe failed: %s", probe.get("error"))

        total_words = sum(r.word_count for r in self._parse_results if r.success)
        total_chunks = len(list(index.docstore.docs.values()))

        self._last_build_stats = {
            "docs": len(documents),
            "chunks": total_chunks,
            "words": total_words,
            "failed": len(files) - len(documents),
            "probe_ok": probe["ok"],
        }

        logger.info(
            "Indexed %d docs, %d chunks, ~%d words in %.1fs (probe=%s)",
            len(documents), total_chunks, total_words, elapsed, probe["ok"]
        )

        await self._sync_documents(docs_dir, files)
        return index

    async def _sync_documents(self, docs_dir: str, files: list[str]) -> None:
        """Update documents table with chunk counts and word counts."""
        db = await get_db()
        await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        await db.execute("DELETE FROM documents")

        # Build lookup: fname → ParseResult (basename for reliable matching)
        result_map = {
            os.path.basename(r.doc.metadata.get("file_name", "") if r.doc else ""): r
            for r in self._parse_results if r.success
        }

        for fname in files:
            fpath = os.path.join(docs_dir, fname)
            try:
                size = os.path.getsize(fpath)
                display_name = await _gemini_clean_filename(fname)
                result = result_map.get(fname)
                word_count = result.word_count if result else 0
                chunk_count = 0

                # Count chunks in index for this file
                if self._index:
                    try:
                        chunks = [
                            n for n in self._index.docstore.docs.values()
                if os.path.basename(n.metadata.get("file_name", "")) == fname
                        ]
                        chunk_count = len(chunks)
                    except Exception:
                        pass

                await db.execute(
                    """INSERT OR REPLACE INTO documents
                       (filename, filepath, file_size, display_name, word_count, chunk_count)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (fname, fpath, size, display_name, word_count, chunk_count),
                )
            except OSError:
                pass

        await db.commit()

    async def _count_documents(self) -> int:
        """Count tracked documents."""
        try:
            db = await get_db()
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
            row = await cursor.fetchone()
            return row["cnt"] if row else 0
        except Exception:
            return 0

    async def get_doc_chunk_count(self, doc_filepath: str) -> int:
        """Check how many chunks exist for this document in the current index."""
        if not self._index:
            return -1  # No index loaded
        try:
            chunks = [
                n for n in self._index.docstore.docs.values()
                if os.path.basename(n.metadata.get("file_name", "")) == doc_filepath
            ]
            return len(chunks)
        except Exception:
            return -1