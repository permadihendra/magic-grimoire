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
        """Handle /docs — list indexed documents with full metadata."""
        from app.plugins.brain.handler import _tool_list_docs_v2
        return await _tool_list_docs_v2()

    async def _handle_index(self, ctx: BotContext) -> str:
        """Handle /index — verify, parse, embed, and report with full diagnostics."""
        global _rag_engine, _doc_indexer

        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            os.makedirs(docs_dir, exist_ok=True)
            return (f"📁 Created `{docs_dir}/` — it's empty!\n\n"
                    "Drop your PDFs or text files there and run `/index` again.")

        files = [
            f for f in os.listdir(docs_dir)
            if os.path.isfile(os.path.join(docs_dir, f))
            and not f.startswith(".")
        ]

        if not files:
            return (f"📭 No files found in `{docs_dir}/`.\n\n"
                    "Place your study documents there first.")

        # ── HTTP helpers ──────────────────────────────────
        import httpx
        token = settings.telegram_token
        chat_id = ctx.chat_id
        send_url = f"https://api.telegram.org/bot{token}/sendMessage"
        edit_url = f"https://api.telegram.org/bot{token}/editMessageText"

        async def _send(text: str) -> int | None:
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(send_url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
                    if r.status_code == 200:
                        return r.json().get("result", {}).get("message_id")
            except Exception:
                pass
            return None

        async def _edit(msg_id: int, text: str):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    r = await client.post(edit_url, json={"chat_id": chat_id, "message_id": msg_id, "text": text, "parse_mode": "Markdown"})
                    if r.status_code >= 400:
                        await client.post(edit_url, json={"chat_id": chat_id, "message_id": msg_id, "text": text})
            except Exception:
                pass

        # ── Phase 1: Pre-index check — test-parse each file ─
        msg_id = await _send(f"🔍 *Pre-index check: {len(files)} file(s)...*")

        pre_lines = [f"🔍 *Pre-index check: {len(files)} file(s)*\n\n"]
        success_count = 0

        for i, fname in enumerate(files, 1):
            fpath = os.path.join(docs_dir, fname)
            try:
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
            except Exception:
                size_mb = 0

            result = await _doc_indexer.pre_index_check_single(fpath, fname)
            if result.success:
                success_count += 1
                pre_lines.append(
                    f"📄 [{i}/{len(files)}] `{fname}` ({size_mb:.1f} MB)\n"
                    f"   └─ {result.method} ✅ — {result.word_count:,} words\n"
                )
            else:
                pre_lines.append(
                    f"📄 [{i}/{len(files)}] `{fname}` ({size_mb:.1f} MB)\n"
                    f"   └─ ❌ {result.error or 'parsing failed'}\n"
                )

        pre_text = "\n".join(pre_lines)

        # Block if ALL files failed parsing
        if success_count == 0:
            logger.error("All %d files failed pre-index check", len(files))
            block_msg = (
                "❌ *No documents could be parsed.*\n\n"
                "All files failed during pre-index check.\n\n"
                "Suggestions:\n"
                "1. `uv sync --extra epub` — install EPUB support\n"
                "2. `uv sync --extra liteparse` — install PDF/OCR support\n"
                "3. Convert EPUB to PDF and re-upload\n"
                "4. Try a plain .txt file\n\n"
                "⚠️ Your old index is preserved. Bot is still functional."
            )
            if msg_id:
                await _edit(msg_id, pre_text + "\n\n" + block_msg)
            return None

        await _edit(msg_id, pre_text + "\n✅ Ready. Building index...")

        # ── Phase 2: Build index with progress ─────────────
        from app.ui.progress import ProgressState, ProgressWatcher
        from app.rag.knowledge_cache import invalidate_cache
        await invalidate_cache()

        progress = ProgressState()
        progress.start_phase("index_embed")
        progress.files_total = len(files)

        watcher = ProgressWatcher(ctx.chat_id, msg_id, progress, interval=30)
        watcher.start()

        t0 = time.time()

        try:
            stats = {}

            def report_chunks(chunk_count: int):
                progress.chunks_embedded = chunk_count

            index = await _doc_indexer.rebuild_index(report_fn=report_chunks)
            _rag_engine.set_index(index)
            stats = _doc_indexer.last_build_stats or {}

        finally:
            progress.complete = True
            watcher.stop()

        elapsed = time.time() - t0

        # ── Phase 3: Post-build report ─────────────────────
        probe_ok = stats.get("probe_ok", False)
        docs_ok = stats.get("docs", 0)
        chunks = stats.get("chunks", 0)
        words = stats.get("words", 0)
        failed = stats.get("failed", 0)
        parse_results = _doc_indexer.parse_results
        parse_errors = _doc_indexer.parse_errors

        # Per-file lines
        file_lines = []
        for fname in files:
            result = next((r for r in parse_results if r.doc and r.doc.metadata.get("file_name") == fname), None)
            error = next((e for e in parse_errors if e[0] == fname), None)
            if result and result.success:
                # Count chunks for this file in the index
                file_chunks = 0
                try:
                    file_chunks = len([n for n in index.docstore.docs.values()
                                       if n.metadata.get("file_name") == fname])
                except Exception:
                    pass
                file_lines.append(
                    f"📄 `{fname}` — ✅ {result.word_count:,} words | {file_chunks} chunks"
                )
            elif error:
                file_lines.append(f"📄 `{fname}` — ❌ {error[1][:60]}")
            else:
                file_lines.append(f"📄 `{fname}` — ❌ unknown error")

        probe_status = "✅ Verified" if probe_ok else "⚠️ Probe failed (index may be empty)"

        result_text = (
            f"✅ *Index built!*\n"
            f"   Docs: {docs_ok} | Chunks: {chunks} | ~{words:,} words ({elapsed:.0f}s)\n"
            f"   {probe_status}\n\n" +
            "\n".join(file_lines) + "\n"
        )

        if failed > 0:
            result_text += f"\n⚠️ {failed} file(s) could not be parsed.\n"

        if chunks == 0:
            result_text += ("\n❌ *WARNING: 0 chunks indexed.*\n"
                            "The index appears empty. Try `/index` again or check file formats.")

        result_text += "\n\nTry `/ask` or `/quiz` to study!"

        if msg_id:
            await _edit(msg_id, result_text)
            return None
        return result_text

    async def _handle_files(self, ctx: BotContext) -> str:
        """Handle /files — list all files on disk with indexed status and metadata."""
        import os

        docs_dir = settings.docs_dir
        if not os.path.isdir(docs_dir):
            return "📁 No files directory found."

        files = []
        for f in os.listdir(docs_dir):
            fpath = os.path.join(docs_dir, f)
            if os.path.isfile(fpath) and not f.startswith("."):
                mtime = os.path.getmtime(fpath)
                files.append((f, fpath, os.path.getsize(fpath), mtime))

        if not files:
            return "📭 No files in docs directory."

        files.sort(key=lambda x: x[3], reverse=True)  # newest first

        # Load indexed state from DB
        from app.database import get_db
        from app.ui.helpers import _build_files_card, _build_total_footer
        db = await get_db()
        db_rows = {}
        try:
            cursor = await db.execute(
                "SELECT filename, display_name, word_count, chunk_count, "
                "       parse_method, verified, indexed_at "
                "FROM documents"
            )
            for r in await cursor.fetchall():
                db_rows[r["filename"]] = dict(r)
        except Exception as e:
            logger.debug("Could not load documents DB: %s", e)

        indexed_count = sum(1 for f in files if f[0] in db_rows)

        lines = [
            f"📁 *Files in docs directory* "
            f"({len(files)} files, {indexed_count} indexed)\n"
        ]
        for i, (fname, fpath, size_bytes, mtime) in enumerate(files, 1):
            db_row = db_rows.get(fname)
            lines.append(_build_files_card(fname, size_bytes, mtime, db_row, i))
            lines.append("")

        lines.append(_build_total_footer(
            total_files=len(files),
            indexed_count=indexed_count,
        ))
        lines.append("_Delete by ID: /delete <number>_ | _Re-index: /index_")
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

        # Load indexed metadata before deleting (for impact report)
        db = await get_db()
        db_row = None
        try:
            cursor = await db.execute(
                "SELECT display_name, word_count, chunk_count, file_size "
                "FROM documents WHERE filename = ?",
                (fname,),
            )
            db_row = await cursor.fetchone()
        except Exception:
            pass

        try:
            os.remove(fpath)
            logger.info("Deleted file #%d: %s", file_id, fpath)

            # Build impact report
            impact = []
            if db_row:
                size_mb = (db_row["file_size"] or 0) / (1024 * 1024)
                words = db_row["word_count"] or 0
                chunks = db_row["chunk_count"] or 0
                if size_mb > 0:
                    impact.append(f"{size_mb:.1f} MB")
                if words > 0:
                    impact.append(f"{words:,} words")
                if chunks > 0:
                    impact.append(f"{chunks} chunks")
                impact_str = " · ".join(impact) if impact else "no metadata"
            else:
                impact_str = "not in index"

            return (
                f"🗑️ *Deleted file #{file_id}:* `{fname}`\n"
                f"   └─ {impact_str}\n\n"
                f"Run `/index` to rebuild the index without this file."
            )

        except OSError as e:
            logger.error("Failed to delete %s: %s", fpath, e)
            return f"⚠️ Failed to delete file: {e}"

            return f"⚠️ Failed to delete file: {e}"

# ── Module-level tool functions (imported by BrainPlugin) ──


async def ask_query(query: str, difficulty: str = "normal", document: str | None = None, chat_id: int | None = None) -> str:
    """Query the document index and return the answer with source citations.

    Args:
        query: Natural language question.
        difficulty: 'simple', 'normal', or 'advanced'.
        document: Optional document name to focus search on.

    Used by BrainPlugin's ask() tool.
    """
    global _rag_engine, _doc_indexer
    import time as _time_module

    _aqid = f"aq_{int(_time_module.time()*1000)%100000:05d}"
    logger.info("[%s] >>> ask_query START q='%s' diff=%s doc=%s", _aqid, query[:60], difficulty, document)
    _aq_t0 = _time_module.time()

    if _rag_engine is None or _doc_indexer is None:
        logger.error("[%s] RAG engine not initialized!", _aqid)
        return "\u26a0\ufe0f RAG engine not initialized. Restart the bot."

    try:
        logger.info("[%s]  calling ensure_index()...", _aqid)
        _idx_t0 = _time_module.time()
        index = await _doc_indexer.ensure_index()
        logger.info("[%s]  index loaded in %.0fms", _aqid, (_time_module.time()-_idx_t0)*1000)
        _rag_engine.set_index(index)

        from app.database import get_db
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        row = await cursor.fetchone()
        if not row or row["cnt"] == 0:
            logger.info("[%s]  no documents indexed", _aqid)
            return "📭 No documents indexed yet! Run `/index` first."

        logger.info("[%s]  calling rag_engine.query_with_sources()...", _aqid)
        _qws_t0 = _time_module.time()
        result = await _rag_engine.query_with_sources(query, difficulty=difficulty, document=document, chat_id=chat_id)
        logger.info("[%s]  q_with_sources done in %.0fms", _aqid, (_time_module.time()-_qws_t0)*1000)
        logger.info("[%s] >>> ask_query DONE (%.0fms total)", _aqid, (_time_module.time()-_aq_t0)*1000)
        return result["answer"]
    except Exception as e:
        logger.error("[%s] >>> ask_query CRASHED: %s", _aqid, e, exc_info=True)
        return f"\u26a0\ufe0f Query failed: {e}"


async def retrieve_passages(query: str, document: str | None = None, chat_id: int | None = None) -> list[dict]:
    """Retrieve passages without LLM generation.

    Args:
        query: Search query.
        document: Optional document name to focus search on.
        chat_id: Telegram chat ID for feedback learning.

    Returns:
        List of dicts with keys: text, filename, score.
    """
    global _rag_engine, _doc_indexer

    if _rag_engine is None or _doc_indexer is None:
        return []

    try:
        index = await _doc_indexer.ensure_index()
        _rag_engine.set_index(index)
        return _rag_engine.retrieve_only(query, document=document, chat_id=chat_id)
    except Exception as e:
        logger.error("retrieve_passages failed: %s", e)
        return []


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

        # Retrieve passages first (for source metadata + token budget)
        passages = _rag_engine.retrieve_only(enhanced_topic)
        chunk_count = len(passages)
        total_tokens = sum(len(p["text"].split()) * 1.3 for p in passages) if passages else 0
        
        result = await _rag_engine.query(enhanced_topic, mode="quiz", difficulty=difficulty, count=count)

        # Append source metadata + next-step (same pattern as /ask answers)
        if passages:
            short_names = []
            seen = set()
            for p in passages[:3]:
                from app.rag.engine import _lookup_display_name
                sn = _lookup_display_name(p["filename"])
                if sn not in seen:
                    seen.add(sn)
                    short_names.append(sn)
            if short_names:
                src_line = "\n\n📖 *Sources used:* " + " · ".join(f"_{s}_" for s in short_names)
                result += src_line

        result += (
            f"\n\n💡 *Next steps:* Ask a follow-up question, or try `/summarize` for a topic overview."
        )
        result += (
            f"\n\n📊 *Retrieval:* {chunk_count} chunks | "
            f"~{int(total_tokens)} tokens | difficulty={difficulty}"
        )

        return result
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
        passages = _rag_engine.retrieve_only(topic)
        chunk_count = len(passages)
        total_tokens = sum(len(p["text"].split()) * 1.3 for p in passages) if passages else 0

        result = await _rag_engine.query(topic, mode="summary")

        if passages:
            short_names = []
            seen = set()
            for p in passages[:3]:
                from app.rag.engine import _lookup_display_name
                sn = _lookup_display_name(p["filename"])
                if sn not in seen:
                    seen.add(sn)
                    short_names.append(sn)
            if short_names:
                src_line = "\n\n📖 *Sources used:* " + " · ".join(f"_{s}_" for s in short_names)
                result += src_line

        result += (
            f"\n\n💡 *Next steps:* Ask `/quiz {topic}` to practice, "
            f"or ask a specific `/ask` question about this topic."
        )
        result += (
            f"\n\n📊 *Retrieval:* {chunk_count} chunks | ~{int(total_tokens)} tokens"
        )

        return result
    except Exception as e:
        logger.error("summarize_topic failed: %s", e, exc_info=True)
        return f"⚠️ Summary failed: {e}"
