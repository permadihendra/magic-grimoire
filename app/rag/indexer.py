"""Document indexer — ingest, chunk, embed, and persist documents.

Documents placed in the configured docs_dir are indexed into a
LlamaIndex VectorStoreIndex and persisted to disk for fast loading.

Uses LiteParse for high-quality parsing (spatial text + OCR + EPUB).
Falls back to SimpleDirectoryReader if LiteParse is not available.
"""

import logging
import os
import time

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage
from llama_index.core import Document as LIDocument

from app.config import settings
from app.database import get_db
from app.rag.models import configure_settings

logger = logging.getLogger(__name__)

# Default persistence directory for the index
INDEX_STORAGE_DIR = os.path.join(
    os.path.dirname(settings.db_path), "index_storage"
)


# ── Parser: LiteParse with fallback ──────────────────────


def _parse_document_raw(raw_bytes: bytes, fname: str) -> LIDocument | None:
    """Parse a document from raw bytes.

    Tries LiteParse first (spatial text + OCR + EPUB support).
    Falls back to SimpleDirectoryReader if LiteParse unavailable.
    Returns a LlamaIndex Document or None on failure.
    """
    # Try 1: LiteParse
    try:
        from liteparse import LiteParse

        # Disable OCR by default — most PDFs have text layers.
        # Install tesseract-ocr + set TESSDATA_PREFIX for OCR support.
        parser = LiteParse(ocr_enabled=False, quiet=True)
        result = parser.parse(raw_bytes)
        text = result.text.strip()
        if text:
            logger.info("LiteParse parsed: %s (%d chars)", fname, len(text))
            return LIDocument(text=text, metadata={"file_name": fname})
        else:
            logger.warning("LiteParse returned empty text for %s", fname)
    except ImportError:
        logger.debug("LiteParse not installed, using fallback")
    except Exception as e:
        logger.warning("LiteParse failed for %s: %s", fname, e)

    # Try 2: SimpleDirectoryReader (reads from file path)
    # Write bytes to temp file for SimpleDirectoryReader
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(fname)[1], delete=False
        ) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        reader = SimpleDirectoryReader(input_files=[tmp_path])
        docs = reader.load_data()

        # Update metadata with original filename
        for d in docs:
            d.metadata["file_name"] = fname

        os.unlink(tmp_path)

        if docs and docs[0].text and docs[0].text.strip():
            logger.info("Fallback parsed: %s (%d chars)", fname, len(docs[0].text))
            return docs[0]
        else:
            logger.warning("Fallback returned empty for %s", fname)
    except Exception as e:
        logger.warning("Fallback parser failed for %s: %s", fname, e)

    return None


# ── DocumentIndexer ──────────────────────────────────────


class DocumentIndexer:
    """Handles document ingestion and index management."""

    def __init__(self) -> None:
        self._index: VectorStoreIndex | None = None

    @property
    def index(self) -> VectorStoreIndex | None:
        return self._index

    async def ensure_index(self) -> VectorStoreIndex:
        """Load existing index or build a new one if none exists.

        Returns the loaded/created VectorStoreIndex.
        """
        if self._index is not None:
            return self._index

        configure_settings()

        if os.path.exists(INDEX_STORAGE_DIR):
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=INDEX_STORAGE_DIR
                )
                self._index = load_index_from_storage(storage_context)
                doc_count = await self._count_documents()
                logger.info("Loaded existing index (%d docs)", doc_count)
                return self._index
            except Exception as e:
                logger.warning("Failed to load existing index: %s", e)

        # Build fresh index
        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            logger.info("Created docs directory: %s", docs_dir)

        self._index = await self._build_index(docs_dir)
        return self._index

    async def rebuild_index(self) -> VectorStoreIndex:
        """Force rebuild the index from scratch."""
        configure_settings()

        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)

        if os.path.exists(INDEX_STORAGE_DIR):
            import shutil
            shutil.rmtree(INDEX_STORAGE_DIR)
            logger.info("Removed old index storage")

        self._index = await self._build_index(docs_dir)
        return self._index

    async def _build_index(self, docs_dir: str) -> VectorStoreIndex:
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

        if not files:
            logger.warning("No documents found in %s — creating empty index", docs_dir)
            from llama_index.core import Document
            dummy = Document(text="")
            index = VectorStoreIndex.from_documents([dummy])
            index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
            self._index = index
            await self._sync_documents(docs_dir, [])
            return index

        start = time.time()
        logger.info("Indexing %d documents from %s...", len(files), docs_dir)

        # Parse each file using LiteParse (or fallback)
        documents = []
        for fname in files:
            fpath = os.path.join(docs_dir, fname)
            try:
                with open(fpath, "rb") as f:
                    raw_bytes = f.read()
                doc = _parse_document_raw(raw_bytes, fname)
                if doc:
                    documents.append(doc)
                else:
                    logger.warning("Skipping unparseable file: %s", fname)
            except Exception as e:
                logger.warning("Failed to read %s: %s", fname, e)

        if not documents:
            logger.error("No documents could be parsed!")
            from llama_index.core import Document
            dummy = Document(text="")
            index = VectorStoreIndex.from_documents([dummy])
            index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)
            self._index = index
            await self._sync_documents(docs_dir, files)
            return index

        logger.info("Parsed %d/%d documents successfully", len(documents), len(files))
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)

        elapsed = time.time() - start
        logger.info("Indexed %d documents in %.1fs", len(files), elapsed)

        await self._sync_documents(docs_dir, files)
        return index

    async def _sync_documents(self, docs_dir: str, files: list[str]) -> None:
        """Update the documents table to reflect current files."""
        db = await get_db()
        await db.execute("DELETE FROM documents")
        for fname in files:
            fpath = os.path.join(docs_dir, fname)
            try:
                size = os.path.getsize(fpath)
                await db.execute(
                    "INSERT OR REPLACE INTO documents (filename, filepath, file_size) "
                    "VALUES (?, ?, ?)",
                    (fname, fpath, size),
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
