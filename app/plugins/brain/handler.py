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
from app.ui.helpers import _split_into_chunks

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
- TOOL: ask(query, difficulty="normal", document=None) — Query your study documents.
  difficulty can be "simple", "normal", or "advanced".
  document: optional document name to focus search on.
  If the user mentions a specific book/document by name, extract it and pass it here.
  The tool output IS the answer (with source citations).
  BEFORE calling this tool, write 1-2 sentences of conversational text explaining 
  what you're about to do. This text will be shown to the user while they wait.
  Example: "Let me look through the World Economy book for key takeaways...\\nTOOL: ask(...)"

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

- TOOL: feedback(type, detail="") — Process user feedback about retrieval quality.
  type: "wrong_source" (user says answer came from wrong book) or 
        "good" (user says answer was helpful/correct) or
        "reset" (clear all feedback for this session).
  detail: describe the document involved or the issue.
  Use for: "that's from the wrong book", "wrong source", "perfect answer",
  "no, I meant the World Economy book not Finance".
  After calling feedback with wrong_source, call ask/quiz again with corrected parameters.

- TOOL: retrieve(query, document=None) — Just find passages without generating text.
  Returns raw passages with filenames and scores. Fast — no LLM generation.
  Use for: quickly checking what documents cover a topic before asking.

Rules:
1. CRITICAL: For ANY question about document content, you MUST call ask(). Do NOT answer from your training data. The ask() output shows "📖 *Primary source:*" if documents were used.
2. NEVER say "I couldn't find" or "it appears" without calling ask() first. Always call ask() — let the RAG engine decide if content exists.
3. ONE ask() call per question. Do NOT split into multiple tools for one question.
4. DOCUMENT DETECTION: When user mentions a book name, pass it as document parameter.
5. DOCUMENT CORRECTION: User says wrong source → feedback(type="wrong_source") + retry with corrected doc.
6. Use conversation context for follow-ups.
7. After tool output, suggest a quiz or deeper explanation.
8. difficulty="simple" for confused, "advanced" for detailed.
9. Chain MULTIPLE tools OK (quiz after ask) but NOT multiple asks for one.
10. Keep text minimal — tools do the heavy lifting.
11. Respond in user's language.
12. NOT FOUND: ask() already handles this. Trust the tool output.

Format for TOOL lines:
TOOL: tool_name(param1="value1", param2=123)

Examples:
User: "hello!"
You: Hey! Ready to study? 📖
TOOL: list_docs()

User: "what is loss aversion?"
You: *(no conversational text before ask — the tool produces the answer)*
TOOL: ask(query="What is loss aversion?", difficulty="normal")

User: "no, that's from the wrong book. It's from Finance for Normal People not World Economy"
You: You're right, let me correct that. *(calls feedback to learn, then retries)*
TOOL: feedback(type="wrong_source", detail="Finance for Normal People")
TOOL: ask(query="what is loss aversion?", document="World Economy")

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
async def _tool_feedback(type: str, detail: str = "", chat_id: int = 0) -> str:
    """Process user feedback about retrieval quality.

    Args:
        type: 'wrong_source' or 'good'.
        detail: Description or document name referenced.
        chat_id: Chat ID for session-scoped learning.

    Returns:
        Confirmation message.
    """
    from app.rag.feedback import get_learner
    learner = get_learner(chat_id)

    if type == "wrong_source":
        # Extract document name from detail if provided
        learner.penalize(detail if detail else "unknown", factor=0.3)
        return (
            f"Got it! I've noted that '{detail}' was the wrong source. "
            "I'll avoid it for this session."
        )
    elif type == "good":
        learner.boost(detail if detail else "all", factor=1.5)
        return (
            f"Great! I've noted that '{detail}' was helpful. "
            "I'll prefer it for this session."
        )
    elif type == "reset":
        learner.clear()
        return "Feedback reset for this session."
    else:
        return f"Unknown feedback type: {type}"


async def _tool_retrieve(query: str, document: str | None = None, chat_id: int = 0) -> list[dict]:
    """Just retrieve passages without LLM generation.
    Returns list of {text, filename, score}.
    """
    from app.plugins.study.handler import retrieve_passages
    return await retrieve_passages(query, document=document, chat_id=chat_id)


async def _tool_ask(query: str, difficulty: str = "normal", document: str | None = None, chat_id: int | None = None) -> str:
    """Execute the ask() tool — RAG query with source citations.

    Args:
        query: The question to answer.
        difficulty: 'simple', 'normal', or 'advanced'.
        document: Optional document name to focus search on.
    """
    from app.plugins.study.handler import ask_query
    try:
        result = await ask_query(query, difficulty=difficulty, document=document, chat_id=chat_id)
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


async def _tool_quiz(topic: str, count: str = "5", difficulty: str = "normal", chat_id: int | None = None) -> str:
    """Execute the quiz() tool — generate practice questions."""
    from app.plugins.study.handler import generate_quiz
    try:
        n = max(1, min(20, int(count)))
    except (ValueError, TypeError):
        n = 5
    try:
        return await generate_quiz(topic, n, difficulty=difficulty, chat_id=chat_id)
    except Exception as e:
        logger.error("quiz() failed: %s", e)
        return f"⚠️ Sorry, quiz generation failed: {e}"


async def _tool_summarize(topic: str, chat_id: int | None = None) -> str:
    """Execute the summarize() tool — topic summary."""
    from app.plugins.study.handler import summarize_topic
    try:
        return await summarize_topic(topic, chat_id=chat_id)
    except Exception as e:
        logger.error("summarize() failed: %s", e)
        return f"⚠️ Sorry, summary failed: {e}"


async def _tool_list_docs_v2() -> str:
    """Execute /docs — list indexed documents with full metadata."""
    from app.database import get_db
    from app.ui.helpers import _build_file_card, _build_total_footer
    try:
        db = await get_db()
        cursor = await db.execute(
            "SELECT filename, file_size, word_count, chunk_count, "
            "       parse_method, verified, indexed_at, display_name "
            "FROM documents ORDER BY indexed_at DESC"
        )
        rows = await cursor.fetchall()
        if not rows:
            return (
                "📭 *No documents indexed yet.*\n\n"
                "Add PDFs/EPUBs to `app/docs/` and run `/index`."
            )

        # Collect totals
        total_words = sum(r["word_count"] or 0 for r in rows)
        total_chunks = sum(r["chunk_count"] or 0 for r in rows)

        lines = [f"📚 *Your Library* ({len(rows)} document{'s' if len(rows) != 1 else ''} indexed)\n"]
        for i, r in enumerate(rows, 1):
            lines.append(_build_file_card(dict(r), index_position=i))
            lines.append("")

        lines.append(_build_total_footer(
            total_docs=len(rows),
            total_words=total_words,
            total_chunks=total_chunks,
        ))

        return "\n".join(lines)
    except Exception as e:
        logger.error("list_docs_v2() failed: %s", e)
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
_SLOW_TOOLS = {"ask", "quiz", "summarize"}


async def _progress_timer(chat_id: int, message_id: int,
                          label: str = "Processing",
                          interval: int = 60):
    """Send 'still working' updates every `interval` seconds.
    Must be cancelled when main work finishes.
    """
    import asyncio
    elapsed = 0
    try:
        while True:
            await asyncio.sleep(interval)
            elapsed += interval
            await _edit_message(chat_id, message_id,
                f"⏳ Still {label}... ({elapsed}s elapsed)")
    except asyncio.CancelledError:
        pass
# ── Telegram message helpers ────────────────────────────

_MAX_TELEGRAM_LENGTH = 4000  # 4096 limit minus safety margin

def _safe_truncate(text: str, max_len: int = _MAX_TELEGRAM_LENGTH) -> str:
    """Truncate text to fit Telegram message limit."""
    if len(text) <= max_len:
        return text
    # Truncate at word boundary
    cut = text[:max_len - 25].rstrip()
    return cut + "\n\n... (truncated)"


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
    "list_docs": _tool_list_docs_v2,
    "chat": _tool_chat,
    "retrieve": _tool_retrieve,
    "feedback": _tool_feedback,
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
            # No tool was called — Gemini answered from training data
            response += ("\n\n"
                "\u2139\ufe0f *This answer came from my training data, not your documents.*\n"
                "For document-sourced answers, use `/ask <question>` "
                "or say \"search your documents for...\""
            )
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

            # Use Gemini's conversational text as thinking message (PLAN audit fix)
            conv_preview = "\n".join(conversational_parts).strip()
            if conv_preview:
                thinking_text = conv_preview
            elif "ask" in tool_names:
                thinking_text = "📖 Searching documents for relevant passages..."
            elif "quiz" in tool_names:
                thinking_text = "📝 Generating practice questions..."
            elif "summarize" in tool_names:
                thinking_text = "📊 Creating topic summary..."
            else:
                thinking_text = "⏳ Processing..."

            thinking_msg = await _send_telegram_message(chat_id, thinking_text)
            thinking_id = thinking_msg.get("message_id") if thinking_msg else None

            # For ask: Phase 1 — retrieve first (fast), show results
            # Create shared state for ProgressWatcher
            from app.ui.progress import ProgressState, ProgressWatcher
            progress = ProgressState()
            progress.query_text = message

            if "ask" in tool_names and thinking_id:
                try:
                    ask_params = next((p for n, p in tool_tasks if n == "ask"), {})
                    query = ask_params.get("query", message)
                    document = ask_params.get("document")

                    progress.start_phase("retrieve")
                    progress.document_name = document

                    passages = await _tool_retrieve(query, document=document, chat_id=chat_id)

                    if passages:
                        short_names = []
                        seen = set()
                        for p in passages[:3]:
                            from app.rag.engine import shorten_filename
                            sn = shorten_filename(p["filename"])
                            if sn not in seen:
                                seen.add(sn)
                                short_names.append(f"{sn} ({p['score']:.2f})")

                        progress.passages_found = len(passages)
                        progress.passages_docs = [p["filename"] for p in passages[:3]]

                        lines = ["🔍 *Retrieved passages:*\n"]
                        for s in short_names:
                            lines.append(f"\u2022 {s}")
                        lines.append("")
                        lines.append("🧠 Generating answer from retrieved passages...")
                        await _edit_message(chat_id, thinking_id, "\n".join(lines))
                except Exception as e:
                    logger.debug("Retrieve preview failed: %s", e)

            # Start progress watcher for long operations
            progress.start_phase("generate")
            watcher = ProgressWatcher(chat_id, thinking_id, progress, interval=60)
            watcher.start()

            # Execute tools with error handling
            result_lines = []
            for t_name, t_params in tool_tasks:
                # Inject chat_id for feedback learning
                t_params = dict(t_params)
                t_params.setdefault("chat_id", chat_id)

                tool_fn = _TOOL_REGISTRY.get(t_name)
                if tool_fn:
                    logger.info("Brain: executing tool '%s' with %s", t_name, t_params)

                    try:
                        result = await tool_fn(**t_params)
                        result_lines.append(result)
                    except Exception as e:
                        logger.error("Tool '%s' failed: %s", t_name, e, exc_info=True)
                        from app.rag.guard import OllamaBusyError, OllamaDeadError
                        if isinstance(e, (OllamaBusyError, OllamaDeadError)):
                            error_msg = str(e)
                        else:
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

            # Stop watcher, mark complete
            progress.complete = True
            watcher.stop()

            conv_text = "\n".join(conversational_parts).strip()
            tool_text = "\n\n".join(result_lines).strip()
            final_reply = f"{conv_text}\n\n{tool_text}" if conv_text else tool_text

            # Truncate if too long for Telegram (4096 char limit)
            final_reply = _safe_truncate(final_reply)

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
