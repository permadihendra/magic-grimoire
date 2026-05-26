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
    commands = ["ask", "quiz", "docs", "index"]
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
        """Handle /docs — list indexed documents."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT filename, file_size, chunk_count, indexed_at FROM documents "
            "ORDER BY indexed_at DESC"
        )
        rows = await cursor.fetchall()

        if not rows:
            return (
                "📭 *No documents indexed.*\n\n"
                "Place your study files in `app/docs/` and run `/index`."
            )

        lines = ["📚 *Indexed Documents*\n"]
        for r in rows:
            size_kb = r["file_size"] / 1024 if r["file_size"] else 0
            chunks = r["chunk_count"] or 0
            lines.append(
                f"📄 `{r['filename']}`\n"
                f"   {size_kb:.0f} KB · {chunks} chunks\n"
            )

        lines.append(f"\n_Total: {len(rows)} documents_")
        return "\n".join(lines)

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
