"""StudyPlugin — RAG-powered study agent for Telegram.

Handles:
- /ask <question> — Query indexed documents
- /quiz <topic> — Generate practice questions
- /docs — List indexed documents
- /index — Re-index documents (admin)
- Free text — Auto-query with RAG

Features:
- Progress updates via Telegram editMessageText during long operations
- Model warm-up on startup to avoid cold-start delays
- Graceful error messages instead of silent failures
"""

import logging
import os
import time

from app.config import settings
from app.database import get_db
from app.plugins.base import BotContext, Plugin
from app.rag.engine import RAGEngine
from app.rag.indexer import DocumentIndexer
from app.rag.models import warm_up

logger = logging.getLogger(__name__)

# Module-level singletons
_rag_engine: RAGEngine | None = None
_doc_indexer: DocumentIndexer | None = None


class StudyPlugin(Plugin):
    name = "study"
    commands = ["ask", "quiz", "docs", "index", "files", "delete"]
    description = "RAG study agent — query documents, generate quizzes"

    async def on_load(self) -> None:
        """Initialize RAG engine and warm up the model."""
        global _rag_engine, _doc_indexer
        _rag_engine = RAGEngine()
        _doc_indexer = DocumentIndexer()
        logger.info("RAG engine initialized (lazy load)")

        # Warm up model in background (non-blocking)
        try:
            await warm_up()
        except Exception as e:
            logger.warning("Model warm-up failed: %s", e)

    # ── Progress helper ──────────────────────────────────────
    async def _progress(self, ctx: BotContext, text: str) -> None:
        """Update the thinking indicator message with a progress update."""
        if not ctx.thinking_msg_id:
            return
        try:
            import httpx

            url = f"https://api.telegram.org/bot{settings.telegram_token}/editMessageText"
            payload = {
                "chat_id": ctx.chat_id,
                "message_id": ctx.thinking_msg_id,
                "text": text,
                "parse_mode": "Markdown",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 400:
                    payload.pop("parse_mode", None)
                    await client.post(url, json=payload)
        except Exception as e:
            logger.debug("Progress update failed (non-critical): %s", e)

    # ── Main handler ─────────────────────────────────────────
    async def handle(self, ctx: BotContext) -> str | None:
        cmd = ctx.message_text.strip().split()[0].split("@")[0].lower()

        if cmd == "/ask":
            return await self._handle_ask(ctx)

        if cmd == "/quiz":
            return await self._handle_quiz(ctx)

        if cmd == "/docs":
            return await self._handle_docs(ctx)

        if cmd == "/index":
            return await self._handle_index(ctx)

        if cmd == "/files":
            return await self._handle_files(ctx)

        if cmd == "/delete":
            return await self._handle_delete(ctx)

        # Free text → auto-query with RAG
        return await self._handle_ask(ctx)

    async def _ensure_index_ready(self, ctx: BotContext) -> str | None:
        """Ensure index is loaded, sending progress updates.

        Returns error message if failed, None on success.
        """
        global _rag_engine, _doc_indexer

        try:
            if _rag_engine is None or _doc_indexer is None:
                return "⚠️ RAG engine not initialized. Restart the bot."

            # Check if we need to build the index
            docs_dir = settings.docs_dir
            if not os.path.isdir(docs_dir):
                os.makedirs(docs_dir, exist_ok=True)
                return (
                    "📭 *No documents to study!*\n\n"
                    f"Place your PDFs/text files in `{docs_dir}/` "
                    "and run `/index` to start."
                )

            # Try loading existing index first (fast path)
            index = await _doc_indexer.ensure_index()
            _rag_engine.set_index(index)

            # Check if we have actual documents
            db = await get_db()
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
            row = await cursor.fetchone()

            if not row or row["cnt"] == 0:
                return (
                    "📭 *No documents indexed yet!*\n\n"
                    "Run `/index` to index your study materials."
                )

            return None  # All good
        except Exception as e:
            logger.error("Failed to ensure index: %s", e, exc_info=True)
            return (
                "⚠️ *Failed to load the document index.*\n\n"
                f"Error: `{e}`\n\n"
                "Make sure Ollama is running (`ollama serve`) and "
                "run `/index` to rebuild."
            )

    async def _handle_ask(self, ctx: BotContext) -> str:
        """Handle /ask <question> or free-text query."""
        text = ctx.message_text.strip()
        question = text

        # Strip /ask prefix if present
        if question.lower().startswith("/ask"):
            question = question[len("/ask"):].strip()
            if not question:
                return (
                    "📝 *Usage:* `/ask <your question>`\n\n"
                    "Example: `/ask What are the key concepts in chapter 3?`\n\n"
                    "Or just send your question directly!"
                )

        await self._progress(ctx, "📖 Loading your study materials...")

        # Ensure index is ready
        error = await self._ensure_index_ready(ctx)
        if error:
            return error

        logger.info("Querying RAG: %s", question[:100])
        await self._progress(ctx, "🔍 Searching documents for relevant passages...")

        try:
            t0 = time.time()
            response = await _rag_engine.query(question, mode="qa")
            elapsed = time.time() - t0

            logger.info("RAG query completed in %.1fs", elapsed)
            return response
        except Exception as e:
            logger.error("RAG query failed: %s", e, exc_info=True)
            err_msg = str(e)
            if "timeout" in err_msg.lower() or "read" in err_msg.lower():
                return (
                    "⚠️ *The AI model took too long to respond.*\n\n"
                    "This often happens on the first query while the model loads into memory.\n"
                    "Please try again — it should be faster this time!\n\n"
                    "If it keeps failing:\n"
                    "• Try a smaller model (`qwen2.5:7b` or `qwen2.5:3b`)\n"
                    "• Check Ollama: `ollama ps`"
                )
            return f"⚠️ *Query failed:* `{err_msg[:300]}`"

    async def _handle_quiz(self, ctx: BotContext) -> str:
        """Handle /quiz <topic> — generate practice questions."""
        text = ctx.message_text.strip()
        topic = text[len("/quiz"):].strip()

        if not topic:
            return (
                "📝 *Usage:* `/quiz <topic>`\n\n"
                "Example: `/quiz Python decorators`\n"
                "Example: `/quiz Chapter 4 — Machine Learning`"
            )

        await self._progress(ctx, "📖 Loading your study materials...")

        # Ensure index is ready
        error = await self._ensure_index_ready(ctx)
        if error:
            return error

        logger.info("Generating quiz on: %s", topic)
        await self._progress(ctx, "📝 Generating practice questions...")

        try:
            t0 = time.time()
            response = await _rag_engine.query(topic, mode="quiz")
            elapsed = time.time() - t0
            logger.info("Quiz generated in %.1fs", elapsed)
            return response
        except Exception as e:
            logger.error("Quiz generation failed: %s", e, exc_info=True)
            return f"⚠️ *Quiz generation failed:* `{str(e)[:300]}`"

    async def _handle_docs(self, ctx: BotContext) -> str:
        """Handle /docs — list indexed documents with professional formatting."""
        from app.plugins.brain.handler import _tool_list_docs
        return await _tool_list_docs()

    async def _handle_index(self, ctx: BotContext) -> str:
        """Handle /index — re-index all documents with progress."""
        global _rag_engine, _doc_indexer

        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            return (
                f"📁 Created `{docs_dir}/` — it's empty!\n\n"
                "Drop your PDFs or text files there and run `/index` again."
            )

        files = [
            f for f in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, f))
            and not f.startswith(".")
        ]

        if not files:
            return (
                f"📭 No files found in `{docs_dir}/`.\n\n"
                "Place your study documents there first."
            )

        await self._progress(
            ctx,
            f"📚 *Indexing {len(files)} documents...*\n\n"
            "This may take 30-60s for the first run.\n"
            "I'll update you as it progresses!",
        )

        logger.info("Rebuilding index with %d files...", len(files))

        try:
            t0 = time.time()

            # Phase 1: Loading
            await self._progress(ctx, "📖 Loading documents...")
            index = await _doc_indexer.rebuild_index()
            _rag_engine.set_index(index)

            elapsed = time.time() - t0

            # Get stats
            db = await get_db()
            cursor = await db.execute("SELECT SUM(chunk_count) as total FROM documents")
            row = await cursor.fetchone()
            total_chunks = row["total"] if row and row["total"] else "?"

            file_list = "\n".join(f"  📄 `{f}`" for f in files)
            return (
                f"✅ *Index rebuilt!* ({elapsed:.0f}s)\n\n"
                f"Indexed `{len(files)}` documents with ~{total_chunks} chunks:\n"
                f"{file_list}\n\n"
                f"Now try `/ask` or `/quiz` to study!"
            )
        except Exception as e:
            logger.error("Index rebuild failed: %s", e, exc_info=True)
            return (
                f"⚠️ *Index rebuild failed:*\n"
                f"`{str(e)[:300]}`\n\n"
                "Make sure Ollama is running (`ollama serve`)."
            )


    async def _handle_files(self, ctx: BotContext) -> str:
        """Handle /files — list all files with numeric IDs."""
        import os
        from datetime import datetime

        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            return "📁 No files directory found."

        files = []
        for f in os.listdir(docs_dir):
            fpath = os.path.join(docs_dir, f)
            if os.path.isfile(fpath) and not f.startswith("."):
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                modified = datetime.fromtimestamp(os.path.getmtime(fpath))
                files.append((f, size_mb, modified))

        if not files:
            return "📭 No files in docs directory."

        files.sort(key=lambda x: x[2], reverse=True)

        # Check which are indexed
        from app.database import get_db
        db = await get_db()
        indexed_files = set()
        try:
            cursor = await db.execute("SELECT filename FROM documents")
            for row in await cursor.fetchall():
                indexed_files.add(row["filename"])
        except Exception:
            pass

        lines = ["📁 *Files in docs directory*\n"]
        for i, (fname, size_mb, modified) in enumerate(files, 1):
            status = "✅ indexed" if fname in indexed_files else "⬜ not indexed"
            time_str = modified.strftime("%d %b %Y, %H:%M")
            # Truncate long filenames for cleaner display
            display_name = fname if len(fname) < 50 else fname[:47] + "..."
            lines.append(
                f"[{i}] `{display_name}`\n"
                f"    └─ {size_mb:.1f} MB · {status} · {time_str}\n"
            )

        lines.append(f"\n_Total: {len(files)} files_")
        lines.append("_Delete by ID_: `/delete <number>`")
        return "\n".join(lines)

    async def _handle_delete(self, ctx: BotContext) -> str:
        """Handle /delete <id> — delete a file by its numeric ID."""
        import os
        from datetime import datetime

        text = ctx.message_text.strip()
        raw = text[len("/delete"):].strip()

        if not raw:
            return (
                "📝 *Usage:* `/delete <id>`\n\n"
                "Delete a file by its number.\n"
                "Use `/files` to see all files with IDs.\n\n"
                "_Example:_ `/delete 3`\n"
                "_Note: Run `/index` after deleting to update the index._"
            )

        # Parse ID
        try:
            file_id = int(raw)
        except ValueError:
            return "⚠️ Please use the file number from `/files`. Example: `/delete 3`"

        if file_id < 1:
            return "⚠️ Invalid ID. Use a positive number from `/files`."

        # Build sorted file list (same order as /files)
        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            return "📁 No files directory found."

        files = []
        for f in os.listdir(docs_dir):
            fpath = os.path.join(docs_dir, f)
            if os.path.isfile(fpath) and not f.startswith("."):
                modified = datetime.fromtimestamp(os.path.getmtime(fpath))
                files.append((f, modified))

        if not files:
            return "📭 No files to delete."

        files.sort(key=lambda x: x[1], reverse=True)

        if file_id > len(files):
            return (
                f"⚠️ Invalid ID `{file_id}`. Use a number between 1 and {len(files)}.\n"
                f"Run `/files` to see the list."
            )

        # Delete by index
        fname = files[file_id - 1][0]
        fpath = os.path.join(docs_dir, fname)

        try:
            os.remove(fpath)
            logger.info("Deleted file #%d: %s", file_id, fpath)
            return (
                f"🗑️ *Deleted file #{file_id}:* `{fname}`\n\n"
                f"Run `/index` to rebuild the index without this file."
            )
        except OSError as e:
            logger.error("Failed to delete %s: %s", fpath, e)
            return f"⚠️ Failed to delete file: {e}"

# ── Module-level tool functions (imported by BrainPlugin) ──


async def ask_query(query: str, difficulty: str = "normal") -> str:
    """Query the document index and return the answer with source citations.

    Args:
        query: Natural language question.
        difficulty: 'simple', 'normal', or 'advanced'.

    Used by BrainPlugin's ask() tool.
    """
    global _rag_engine, _doc_indexer

    if _rag_engine is None or _doc_indexer is None:
        return "⚠️ RAG engine not initialized. Restart the bot."

    try:
        index = await _doc_indexer.ensure_index()
        _rag_engine.set_index(index)

        # Check if we have documents
        from app.database import get_db
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        row = await cursor.fetchone()
        if not row or row["cnt"] == 0:
            return "📭 No documents indexed yet! Run `/index` first."

        logger.info("ask_query: %s (difficulty=%s)", query[:100], difficulty)
        result = await _rag_engine.query_with_sources(query, difficulty=difficulty)
        return result["answer"]
    except Exception as e:
        logger.error("ask_query failed: %s", e, exc_info=True)
        return f"⚠️ Query failed: {e}"


async def generate_quiz(topic: str, count: int = 5, difficulty: str = "normal") -> str:
    """Generate practice questions on a topic.

    Args:
        topic: Subject to quiz on.
        count: Number of questions (1-20).
        difficulty: 'simple', 'normal', or 'advanced'.

    Used by BrainPlugin's quiz() tool.
    """
    global _rag_engine, _doc_indexer

    if _rag_engine is None or _doc_indexer is None:
        return "⚠️ RAG engine not initialized. Restart the bot."

    try:
        index = await _doc_indexer.ensure_index()
        _rag_engine.set_index(index)

        from app.database import get_db
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        row = await cursor.fetchone()
        if not row or row["cnt"] == 0:
            return "📭 No documents indexed yet! Run `/index` first."

        logger.info("generate_quiz: %s (count=%d, difficulty=%s)", topic[:80], count, difficulty)
        enhanced_topic = f"Generate {count} practice questions about: {topic}"
        return await _rag_engine.query(enhanced_topic, mode="quiz", difficulty=difficulty, count=count)
    except Exception as e:
        logger.error("generate_quiz failed: %s", e, exc_info=True)
        return f"⚠️ Quiz generation failed: {e}"


async def summarize_topic(topic: str) -> str:
    """Generate a summary of a topic.

    Used by BrainPlugin's summarize() tool.
    """
    global _rag_engine, _doc_indexer

    if _rag_engine is None or _doc_indexer is None:
        return "⚠️ RAG engine not initialized. Restart the bot."

    try:
        index = await _doc_indexer.ensure_index()
        _rag_engine.set_index(index)

        from app.database import get_db
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        row = await cursor.fetchone()
        if not row or row["cnt"] == 0:
            return "📭 No documents indexed yet! Run `/index` first."

        logger.info("summarize_topic: %s", topic[:80])
        return await _rag_engine.query(topic, mode="summary")
    except Exception as e:
        logger.error("summarize_topic failed: %s", e, exc_info=True)
        return f"⚠️ Summary failed: {e}"
