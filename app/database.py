import logging
import os

import aiosqlite

from app.config import settings

logger = logging.getLogger(__name__)

_db_connection: aiosqlite.Connection | None = None


async def get_db() -> aiosqlite.Connection:
    """Get the shared aiosqlite connection, creating it on first call.
    Auto-recovers from corruption caused by hard crashes."""
    global _db_connection
    if _db_connection is not None:
        # Verify existing connection is still healthy
        try:
            await _db_connection.execute("SELECT 1")
            return _db_connection
        except Exception:
            logger.warning("DB connection lost — reconnecting")
            _db_connection = None

    os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
    
    # Check for corruption before connecting
    if os.path.exists(settings.db_path):
        import sqlite3
        try:
            test = sqlite3.connect(settings.db_path)
            test.execute("PRAGMA quick_check")
            test.close()
        except sqlite3.DatabaseError:
            logger.error("Database corrupted — recreating from scratch")
            _recover_db()

    _db_connection = await aiosqlite.connect(settings.db_path)
    _db_connection.row_factory = aiosqlite.Row
    await _db_connection.execute("PRAGMA journal_mode=WAL")
    await _db_connection.execute("PRAGMA foreign_keys=ON")
    await _db_connection.execute("PRAGMA synchronous=NORMAL")
    return _db_connection


def _recover_db() -> None:
    """Backup corrupted DB and delete it for fresh recreation."""
    import shutil, time
    bak = f"{settings.db_path}.corrupted.{int(time.time())}"
    try:
        shutil.copy2(settings.db_path, bak)
        logger.info("Corrupted DB backed up to %s", bak)
    except Exception:
        pass
    os.remove(settings.db_path)
    logger.info("Corrupted DB deleted — will be recreated on next start")


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
            display_name TEXT,
            indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add display_name if upgrading from older schema
    try:
        await db.execute("ALTER TABLE documents ADD COLUMN display_name TEXT")
    except Exception:
        pass  # Column already exists

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

    # Knowledge cache — Q&A pairs for fast repeat queries
    await db.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            question_embedding BLOB,
            short_answer TEXT NOT NULL,
            full_answer TEXT NOT NULL,
            source_doc TEXT,
            retrieval_score REAL DEFAULT 0.0,
            approved INTEGER DEFAULT 0,
            used_count INTEGER DEFAULT 0,
            chat_id INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_used_at DATETIME
        )
    """)

    # FTS5 full-text search on document names
    await db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            filename,
            display_name,
            content=documents,
            content_rowid=id
        )
    """)

    # Triggers to keep FTS5 in sync
    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS docs_fts_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, filename, display_name)
            VALUES (new.id, new.filename, new.display_name);
        END
    """)
    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS docs_fts_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, filename, display_name)
            VALUES ('delete', old.id, old.filename, old.display_name);
        END
    """)
    await db.execute("""
        CREATE TRIGGER IF NOT EXISTS docs_fts_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, filename, display_name)
            VALUES ('delete', old.id, old.filename, old.display_name);
            INSERT INTO documents_fts(rowid, filename, display_name)
            VALUES (new.id, new.filename, new.display_name);
        END
    """)

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
