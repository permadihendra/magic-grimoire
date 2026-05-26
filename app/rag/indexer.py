"""Document indexer — ingest, chunk, embed, and persist documents.

Documents placed in the configured docs_dir are indexed into a
LlamaIndex VectorStoreIndex and persisted to disk for fast loading.
"""

import logging
import os
import time

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core import load_index_from_storage

from app.config import settings
from app.database import get_db
from app.rag.models import configure_settings

logger = logging.getLogger(__name__)

# Default persistence directory for the index
INDEX_STORAGE_DIR = os.path.join(
    os.path.dirname(settings.db_path), "index_storage"
)


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
                # Fall through to rebuild

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
            logger.info("Created docs directory: %s", docs_dir)

        # Remove old index if exists
        if os.path.exists(INDEX_STORAGE_DIR):
            import shutil
            shutil.rmtree(INDEX_STORAGE_DIR)
            logger.info("Removed old index storage")

        self._index = await self._build_index(docs_dir)
        return self._index

    async def _build_index(self, docs_dir: str) -> VectorStoreIndex:
        """Build a new index from documents in the given directory."""
        # Supported document extensions
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

        # Explicitly pass only valid document file paths
        file_paths = [os.path.join(docs_dir, f) for f in files]
        documents = SimpleDirectoryReader(
            input_files=file_paths,
            filename_as_id=True,
            file_metadata=lambda fn: {"file_name": os.path.basename(fn)},
        ).load_data()

        # Filter out any empty docs
        documents = [d for d in documents if d.text and d.text.strip()]
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        index.storage_context.persist(persist_dir=INDEX_STORAGE_DIR)

        elapsed = time.time() - start
        logger.info("Indexed %d documents in %.1fs", len(files), elapsed)

        # Track in SQLite
        await self._sync_documents(docs_dir, files)

        return index

    async def _sync_documents(self, docs_dir: str, files: list[str]) -> None:
        """Update the documents table to reflect current files."""
        db = await get_db()
        # Clear old entries
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
