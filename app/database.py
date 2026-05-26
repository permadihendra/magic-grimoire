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
