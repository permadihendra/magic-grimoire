"""Database module — fresh sqlite3 connection per operation.

Each get_db() call opens a new connection. No caching, no background threads.
This prevents corruption during GPU-heavy operations (embedding 1264 chunks).
"""

import logging
import os
import sqlite3
import threading

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class _AsyncCursor:
    """Async wrapper around sqlite3.Cursor."""
    def __init__(self, cur: sqlite3.Cursor):
        self._cur = cur

    async def fetchone(self):
        return self._cur.fetchone()

    async def fetchall(self):
        return self._cur.fetchall()


class _AsyncDB:
    """Thin async wrapper — opens fresh sqlite3 connection each time."""

    def __init__(self):
        os.makedirs(os.path.dirname(settings.db_path), exist_ok=True)
        self._conn = sqlite3.connect(settings.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    async def execute(self, sql: str, parameters=None):
        with _lock:
            if parameters:
                return _AsyncCursor(self._conn.execute(sql, parameters))
            return _AsyncCursor(self._conn.execute(sql))

    async def commit(self):
        with _lock:
            self._conn.commit()


async def get_db():
    """Get a fresh async-wrapped database connection."""
    return _AsyncDB()


async def init_db() -> None:
    db = await get_db()
async def init_db() -> None:
    db = await get_db()

    # Build the documents table — idempotent via separate ALTERs (handles existing tables)
    await db.execute(
        "CREATE TABLE IF NOT EXISTS documents ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL, "
        "  filepath TEXT NOT NULL UNIQUE, file_size INTEGER DEFAULT 0, "
        "  display_name TEXT, indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP"
        ")",
    )
    # Add columns idempotently (OK if they already exist)
    for col, coltype in [
        ("display_name", "TEXT"),
        ("word_count", "INTEGER DEFAULT 0"),
        ("chunk_count", "INTEGER DEFAULT 0"),
        ("parse_method", "TEXT"),
        ("verified", "INTEGER DEFAULT 0"),
    ]:
        try:
            await db.execute(f"ALTER TABLE documents ADD COLUMN {col} {coltype}")
            logger.info("Added column: %s", col)
        except Exception:
            pass
    await db.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            response TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
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
    await db.commit()
    logger.info("Database initialized")


async def get_document_stats() -> dict:
    try:
        db = await get_db()
        cur = await db.execute(
            "SELECT COUNT(*) as doc_count, COALESCE(SUM(file_size), 0) as total_bytes "
            "FROM documents"
        )
        row = await cur.fetchone()
        if not row:
            return {"docs": 0, "size_mb": 0, "words": 0, "chunks": 0}
        size_mb = round(row["total_bytes"] / (1024 * 1024), 1)
        words = int(row["total_bytes"] * 0.35 / 5) if row["total_bytes"] else 0
        return {"docs": row["doc_count"], "size_mb": size_mb, "words": words, "chunks": 0}
    except Exception:
        return {"docs": 0, "size_mb": 0, "words": 0, "chunks": 0}
