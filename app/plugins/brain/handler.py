"""BrainPlugin — Gemini agent that routes intents to tools.

Intercepts all non-command messages, sends them to Gemini with
tool definitions, and executes the chosen tools.

Features:
- Tool-based agentic flow (Option C)
- Conversation memory (last 5 exchanges per chat)
- Proactive suggestions after answers
- Source citations when answering
- Difficulty levels (simple/normal/advanced)
"""

import logging
import re
from collections import defaultdict

from app.config import settings
from app.llm.gemini import gemini_chat
from app.plugins.base import BotContext, Plugin

logger = logging.getLogger(__name__)

# ── Conversation memory ──────────────────────────────────
# {chat_id: [(user_msg, bot_msg), ...]} — last 5 exchanges
_conversation_memory: dict[int, list[tuple[str, str]]] = defaultdict(list)
MAX_HISTORY = 5


def _build_history_block(chat_id: int) -> str:
    """Build a conversation history string for Gemini context."""
    history = _conversation_memory.get(chat_id, [])
    if not history:
        return ""

    lines = ["\n--- Recent conversation context ---"]
    for i, (user_msg, bot_msg) in enumerate(history[-MAX_HISTORY:], 1):
        lines.append(f"[{i}] User: {user_msg[:200]}")
        lines.append(f"[{i}] You: {bot_msg[:200]}")
    lines.append("--- End of context ---\n")
    return "\n".join(lines)


def _remember(chat_id: int, user_msg: str, bot_msg: str) -> None:
    """Store an exchange in conversation memory."""
    history = _conversation_memory[chat_id]
    history.append((user_msg, bot_msg))
    if len(history) > MAX_HISTORY:
        _conversation_memory[chat_id] = history[-MAX_HISTORY:]


# ── Agent system prompt ───────────────────────────────────
AGENT_PROMPT = """You are Magic Grimoire, a study assistant agent that helps users learn from their documents. You are BOTH a tutor and a librarian.

As a TUTOR:
- Teach concepts from the user's OWN documents, not from your training data
- Adapt to the user's level (beginner → advanced)
- After the tool output, suggest one relevant next step
- Check if the user wants a quiz or deeper explanation

As a LIBRARIAN:
- The ask() tool automatically cites source documents
- Know what documents are available — use list_docs() when asked
- Recommend relevant documents for the user's questions

Available tools:
- TOOL: ask(query, difficulty="normal") — Query your study documents.
  difficulty can be "simple", "normal", or "advanced".
  The tool output IS the answer (with source citations).
  Do NOT write your own answer before this tool.

- TOOL: quiz(topic, count=5, difficulty="normal") — Generate practice questions.
  The tool output IS the quiz.
  Do NOT write your own questions before this tool.

- TOOL: summarize(topic) — Create a topic summary.
  The tool output IS the summary.
  Do NOT write your own summary before this tool.

- TOOL: list_docs() — List all indexed documents with details.
  Use for: "what docs do you have?", "show my documents".

- TOOL: chat(text) — Casual conversation. No document needed.
  Use for: greetings, thanks, feedback, casual chat.

Rules:
1. CRITICAL: For ask/quiz/summarize tools — do NOT write any conversational text or answers BEFORE the tool. Let the tool produce the complete answer. You may add ONE short line AFTER the tool (a suggestion or follow-up question).
2. Use conversation context for follow-ups like "tell me more", "explain that".
3. After the tool output, naturally suggest: a quiz, a related topic, or deeper explanation.
4. If user seems confused, use difficulty="simple".
5. If user asks for advanced/detailed content, use difficulty="advanced".
6. You can chain MULTIPLE tools in one response.
7. Keep your text minimal — the tools do the heavy lifting.
8. Respond in the user's language (Indonesian or English).
9. NOT FOUND HANDLING: If the tool returns empty, "not found", or a very short answer — acknowledge it honestly. Don't pretend you found something. Suggest: different keywords, upload relevant docs, or check /files.

Format for TOOL lines:
TOOL: tool_name(param1="value1", param2=123)

Examples:
User: "hello!"
You: Hey! Ready to study? 📖
TOOL: list_docs()

User: "what is loss aversion?"
You: *(no conversational text before ask — the tool produces the answer)*
TOOL: ask(query="What is loss aversion?", difficulty="normal")

User: "explain it simply"
TOOL: ask(query="What is loss aversion? Explain simply", difficulty="simple")

User: "quiz me on chapter 3"
TOOL: quiz(topic="chapter 3", count=5, difficulty="normal")

User: "too easy, make it harder"
TOOL: quiz(topic="chapter 3", count=5, difficulty="advanced")
"""

# ── TOOL regex ────────────────────────────────────────────
_TOOL_RE = re.compile(
    r'TOOL:\s*(\w+)\(([^)]*)\)',
    re.IGNORECASE,
)


def _parse_tool(line: str) -> tuple[str, dict] | None:
    """Parse a TOOL: line into (tool_name, params_dict) or None."""
    m = _TOOL_RE.search(line)
    if not m:
        return None

    name = m.group(1).strip().lower()
    raw_params = m.group(2).strip()

    params: dict = {}
    if raw_params:
        for part in raw_params.split(","):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                params[key] = val

    return name, params


# ── Tool implementations ──────────────────────────────────
async def _tool_ask(query: str, difficulty: str = "normal") -> str:
    """Execute the ask() tool — RAG query with source citations."""
    from app.plugins.study.handler import ask_query
    try:
        result = await ask_query(query, difficulty=difficulty)
        # Validate: empty or too-short response
        if not result or len(result.strip()) < 20:
            return (
                "📭 *I searched your documents but couldn't find a good answer.*\n\n"
                "Suggestions:\n"
                "• Try different keywords\n"
                "• Upload more relevant documents\n"
                "• Use `/files` to check your available docs\n"
                "• Run `/index` if you recently added files"
            )
        return result
    except Exception as e:
        logger.error("ask() failed: %s", e)
        return f"⚠️ Sorry, I couldn't find an answer.\n\nError: {e}"


async def _tool_quiz(topic: str, count: str = "5", difficulty: str = "normal") -> str:
    """Execute the quiz() tool — generate practice questions."""
    from app.plugins.study.handler import generate_quiz
    try:
        n = max(1, min(20, int(count)))
    except (ValueError, TypeError):
        n = 5
    try:
        return await generate_quiz(topic, n, difficulty=difficulty)
    except Exception as e:
        logger.error("quiz() failed: %s", e)
        return f"⚠️ Sorry, quiz generation failed: {e}"


async def _tool_summarize(topic: str) -> str:
    """Execute the summarize() tool — topic summary."""
    from app.plugins.study.handler import summarize_topic
    try:
        return await summarize_topic(topic)
    except Exception as e:
        logger.error("summarize() failed: %s", e)
        return f"⚠️ Sorry, summary failed: {e}"


async def _tool_list_docs() -> str:
    """Execute the list_docs() tool — list indexed documents."""
    from app.database import get_db
    try:
        db = await get_db()
        cursor = await db.execute(
            "SELECT filename, file_size, chunk_count, indexed_at FROM documents "
            "ORDER BY indexed_at DESC"
        )
        rows = await cursor.fetchall()
        if not rows:
            return "📭 *No documents indexed yet.*\n\nAdd PDFs to `app/docs/` and run `/index`."

        lines = ["📚 *Indexed Documents*\n"]
        for r in rows:
            size_mb = r["file_size"] / (1024 * 1024) if r["file_size"] else 0
            chunks = r["chunk_count"] or 0
            indexed = r["indexed_at"] or "unknown"

            # Format timestamp nicely
            try:
                from datetime import datetime
                dt = datetime.strptime(indexed, "%Y-%m-%d %H:%M:%S")
                indexed_fmt = dt.strftime("%d %b %Y, %H:%M")
            except (ValueError, TypeError):
                indexed_fmt = str(indexed)

            name = r["filename"].replace(".pdf", "").replace("-", " ").replace("_", " ")
            lines.append(f"📄 **{name}**")
            lines.append(f"   └─ {size_mb:.1f} MB · {chunks} chunks · indexed {indexed_fmt}")
            lines.append("")

        lines.append(f"_Total: {len(rows)} documents_")
        return "\n".join(lines)
    except Exception as e:
        logger.error("list_docs() failed: %s", e)
        return f"⚠️ Failed to list documents: {e}"


async def _tool_chat(text: str) -> str:
    """Execute the chat() tool — direct Ollama conversation (no RAG)."""
    from app.rag.models import get_llm
    try:
        llm = get_llm()
        personality = settings.ai_personality.strip()
        prompt = (
            f"Personality: {personality}\n\nRespond conversationally to: {text}"
            if personality
            else f"Respond conversationally to: {text}"
        )
        response = await llm.acomplete(prompt)
        return str(response).strip()
    except Exception as e:
        logger.error("chat() failed: %s", e)
        return "Hey! 😊 I'm here to help you study. Ask me anything about your documents!"


# ── Slow tools (need thinking indicator) ─────────────────
# These take >5s due to Ollama LLM generation
_SLOW_TOOLS = {"ask", "quiz", "summarize"}


async def _send_telegram_message(chat_id: int, text: str) -> dict | None:
    """Send a Telegram message and return the result (with message_id)."""
    import httpx
    url = f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 400:
            payload.pop("parse_mode", None)
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            return resp.json().get("result")
    return None


async def _edit_message(chat_id: int, message_id: int, text: str) -> bool:
    """Edit a Telegram message. Returns True on success."""
    import httpx
    url = f"https://api.telegram.org/bot{settings.telegram_token}/editMessageText"
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text, "parse_mode": "Markdown"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 400:
                payload.pop("parse_mode", None)
                resp = await client.post(url, json=payload)
            return resp.status_code == 200
    except Exception:
        return False


# ── Tool registry ─────────────────────────────────────────
_TOOL_REGISTRY = {
    "ask": _tool_ask,
    "quiz": _tool_quiz,
    "summarize": _tool_summarize,
    "list_docs": _tool_list_docs,
    "chat": _tool_chat,
}


# ── BrainPlugin ───────────────────────────────────────────
class BrainPlugin(Plugin):
    name = "brain"
    commands = []
    description = "AI agent that understands your intent and routes to the right tool"

    async def handle(self, ctx: BotContext) -> str | None:
        message = ctx.message_text.strip()
        chat_id = ctx.chat_id

        # ── 1. Build prompt with conversation history ─────
        history_block = _build_history_block(chat_id)
        full_prompt = AGENT_PROMPT
        if history_block:
            # Prepend history to user message, not system prompt
            user_message = f"{history_block}\nCurrent message: {message}"
        else:
            user_message = message

        logger.info("Brain: sending to Gemini: %s", message[:80])

        try:
            response = await gemini_chat(
                system_prompt=full_prompt,
                user_message=user_message,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
            )
        except RuntimeError as e:
            logger.error("Gemini call failed: %s", e)
            reply = await _tool_chat(
                f"The user said: {message}. Respond conversationally."
            )
            _remember(chat_id, message, reply)
            return reply

        logger.info("Brain: Gemini response (first 100): %s", response[:100])

        # ── 2. Check for TOOL: lines ─────────────────────
        if "TOOL:" not in response:
            _remember(chat_id, message, response)
            return response  # Fast path — gateway sends reply

        # ── 3. Parse tools from response ────────────────────
        lines = response.split("\n")
        tool_tasks = []
        conversational_parts = []

        for line in lines:
            if "TOOL:" in line:
                parsed = _parse_tool(line)
                if parsed:
                    tool_tasks.append(parsed)
            else:
                if line.strip():
                    conversational_parts.append(line)

        is_slow = any(t in _SLOW_TOOLS for t, _ in tool_tasks)

        # ── 4a. Slow path — send thinking, execute, edit ────
        if is_slow:
            tool_names = [t for t, _ in tool_tasks]

            # Send initial thinking message
            if "ask" in tool_names:
                thinking_text = "📖 Searching documents for relevant passages..."
            elif "quiz" in tool_names:
                thinking_text = "📝 Generating practice questions..."
            elif "summarize" in tool_names:
                thinking_text = "📊 Creating topic summary..."
            else:
                thinking_text = "⏳ Processing..."

            thinking_msg = await _send_telegram_message(chat_id, thinking_text)
            thinking_id = thinking_msg.get("message_id") if thinking_msg else None

            # Commitment: show stats + time estimate for large queries
            if thinking_id and ("ask" in tool_names or "summarize" in tool_names):
                try:
                    from app.database import get_document_stats
                    stats = await get_document_stats()
                    if stats and stats["docs"] > 0 and stats["words"] > 20_000:
                        w = stats["words"]
                        d = stats["docs"]
                        if w < 50_000:
                            estimate = "~1 min"
                        elif w < 200_000:
                            estimate = "~2-3 min"
                        elif w < 500_000:
                            estimate = "~3-5 min"
                        elif w < 1_000_000:
                            estimate = "~5-10 min"
                        else:
                            estimate = "~10+ min"

                        await _edit_message(chat_id, thinking_id,
                            f"📍 Found passages across **{d} documents**\n"
                            f"📚 Corpus: **{w//1000}K** estimated words\n"
                            f"⏳ Analyzing and formulating your answer... {estimate}\n\n"
                            f"I'll notify you as soon as it's ready! ✅")
                except Exception as e:
                    logger.debug("Stats/estimate update failed: %s", e)

            # Execute tools with error handling
            result_lines = []
            for t_name, t_params in tool_tasks:
                tool_fn = _TOOL_REGISTRY.get(t_name)
                if tool_fn:
                    logger.info("Brain: executing tool '%s' with %s", t_name, t_params)
                    try:
                        result = await tool_fn(**t_params)
                        result_lines.append(result)
                    except Exception as e:
                        logger.error("Tool '%s' failed: %s", t_name, e, exc_info=True)
                        error_msg = (
                            "⚠️ *Sorry, the analysis failed.*\n\n"
                            f"Error: `{e}`\n\n"
                            "Suggestions:\n"
                            "• Try a simpler or more specific question\n"
                            "• Check that Ollama is running (`ollama serve`)\n"
                            "• Run `/index` to rebuild the index"
                        )
                        if thinking_id:
                            await _edit_message(chat_id, thinking_id, error_msg)
                        _remember(chat_id, message, error_msg)
                        return None
                else:
                    result_lines.append(f"⚠️ Unknown tool: {t_name}")

            conv_text = "\n".join(conversational_parts).strip()
            tool_text = "\n\n".join(result_lines).strip()
            final_reply = f"{conv_text}\n\n{tool_text}" if conv_text else tool_text

            if thinking_id:
                ok = await _edit_message(chat_id, thinking_id, final_reply)
                if not ok:
                    _remember(chat_id, message, final_reply)
                    return final_reply

            _remember(chat_id, message, final_reply)
            return None

        # ── 4b. Fast path — execute directly, return text ───
        result_lines = []
        for t_name, t_params in tool_tasks:
            tool_fn = _TOOL_REGISTRY.get(t_name)
            if tool_fn:
                result = await tool_fn(**t_params)
                result_lines.append(result)
            else:
                result_lines.append(f"⚠️ Unknown tool: {t_name}")

        conv_text = "\n".join(conversational_parts).strip()
        tool_text = "\n\n".join(result_lines).strip()
        final_reply = f"{conv_text}\n\n{tool_text}" if conv_text else tool_text
        _remember(chat_id, message, final_reply)
        return final_reply  # Fast path — gateway sends reply
