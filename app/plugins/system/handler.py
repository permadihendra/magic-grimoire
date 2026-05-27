from app.plugins.base import BotContext, Plugin, PluginRegistry


class SystemPlugin(Plugin):
    name = "system"
    commands = ["start", "help", "ping", "status"]
    description = "Basic bot commands — ping, help, status"

    async def handle(self, ctx: BotContext) -> str | None:
        cmd = ctx.message_text.strip().split()[0].split("@")[0].lower()

        if cmd == "/ping":
            return "🏓 pong"

        if cmd == "/start":
            return (
                "👋 Welcome to *Magic Grimoire* 📖✨\n\n"
                "Your personal RAG study agent. I can answer questions "
                "from your indexed documents, generate quizzes, and help "
                "you prepare for exams.\n\n"
                "Send `/help` to see what I can do."
            )

        if cmd == "/help":
            return self._build_help()

        if cmd == "/status":
            return await self._build_status(ctx)

        if cmd == "/summarize":
            topic = ctx.message_text.strip()[len("/summarize"):].strip()
            if not topic:
                return (
                    "📝 *Usage:* `/summarize <topic>`\n\n"
                    "Example: `/summarize the main themes of this chapter`"
                )
            study = PluginRegistry.get().get_plugin("study")
            if study:
                result = await study.handle(ctx)
                return result if isinstance(result, str) else result.reply
            return "⚠️ Study plugin unavailable."

        return None

    async def _build_status(self, ctx: BotContext) -> str:
        """Show index and bot status."""
        index_stats = "📭 No documents indexed yet."
        try:
            from app.database import get_db
            db = await get_db()
            cursor = await db.execute(
                "SELECT COUNT(*) as docs, COALESCE(SUM(chunk_count), 0) as chunks FROM documents"
            )
            row = await cursor.fetchone()
            if row and row["docs"] > 0:
                index_stats = (
                    f"📚 *{row['docs']}* documents indexed\n"
                    f"🧩 *{row['chunks']}* total chunks"
                )
        except Exception:
            pass

        return (
            "📊 *Magic Grimoire Status*\n\n"
            "✅ Bot running\n"
            f"{index_stats}\n"
            f"🔗 Chat ID: `{ctx.chat_id}`\n\n"
            "⚡ Powered by Ollama + Qwen3:4b"
        )

    def _build_help(self) -> str:
        return (
            "📖 *Magic Grimoire — Commands*\n\n"
            "📚 *Study & Query*\n"
            "`/ask <question>`\n"
            "  └─ Query your indexed documents\n"
            "`/quiz <topic>`\n"
            "  └─ Generate practice questions\n"
            "`/summarize <topic>`\n"
            "  └─ Create a topic summary\n\n"
            "📂 *Documents*\n"
            "`/docs`\n"
            "  └─ List indexed documents\n"
            "`/files`\n"
            "  └─ List all files on disk\n"
            "`/delete <name>`\n"
            "  └─ Delete a file\n"
            "`/index`\n"
            "  └─ Re-index all documents\n\n"
            "⚙️ *System*\n"
            "`/ping` — Health check\n"
            "`/status` — Bot + index status\n"
            "`/help` — This message\n\n"
            "💡 *Free text:*\n"
            "Just send any question — I'll search your docs automatically!"
        )