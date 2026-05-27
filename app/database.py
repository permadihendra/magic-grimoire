import logging
import os

import aiosqlite

from app.config import settings

logger = logging.getLogger(__name__)

_db_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """Get the shared aiosqlite connection, creating it on first call."""
    global _db_connection
    if _db_connection is None:
        os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
        _db_connection = await aiosqlite.connect(settings.db_path)
        _db_connection.row_factory = aiosqlite.Row
        await _db_connection.execute("PRAGMA journal_mode=WAL")
        await _db_connection.execute("PRAGMA foreign_keys=ON")
        await _db_connection.execute("PRAGMA synchronous=NORMAL")
    return _db_connection


async def init_db() -> None:
    """Initialize database tables."""
    db = await get_db()

    # Documents tracking table
    await db.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL UNIQUE,
            file_size INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Query history (for future context)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            response TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    await db.commit()
    logger.info("Database initialized")


async def get_document_stats() -> dict:
    """Get aggregate stats about indexed documents.

    Returns:
        Dict with keys: docs (count), size_mb (total MB), words (estimated), chunks.
        Returns zeros if no documents are indexed.
    """
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT COUNT(*) as doc_count, COALESCE(SUM(file_size), 0) as total_bytes, "
            "COALESCE(SUM(chunk_count), 0) as total_chunks FROM documents"
        )
        row = await cursor.fetchone()
        if not row or row["doc_count"] == 0:
            return {"docs": 0, "size_mb": 0, "words": 0, "chunks": 0}

        doc_count = row["doc_count"]
        total_bytes = row["total_bytes"]
        total_chunks = row["total_chunks"]

        # Estimate word count: rough heuristic
        # PDF/EPUB: ~35% text content, ~5 chars per word
        if total_chunks > 0:
            # Better estimate from chunk count * chunk size
            from app.config import settings
            estimated_words = total_chunks * settings.chunk_size // 5
        else:
            estimated_words = int(total_bytes * 0.35 / 5)

        return {
            "docs": doc_count,
            "size_mb": round(total_bytes / (1024 * 1024), 1),
            "words": estimated_words,
            "chunks": total_chunks,
        }
    except Exception as e:
        logger.warning("get_document_stats failed: %s", e)
        return {"docs": 0, "size_mb": 0, "words": 0, "chunks": 0}
