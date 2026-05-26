"""BrainPlugin — Gemini agent that routes intents to tools.

Intercepts all non-command messages, sends them to Gemini with
tool definitions, and executes the chosen tools.

This is Option C: Tool-based agentic flow.
Gemini decides which tools to call (and can chain multiple).
"""

import logging
import re

from app.config import settings
from app.llm.gemini import gemini_chat
from app.plugins.base import BotContext, Plugin

logger = logging.getLogger(__name__)

# ── Agent system prompt ───────────────────────────────────
AGENT_PROMPT = """You are Magic Grimoire, a study assistant agent that helps users learn from their documents. You have tools available — use them when appropriate.

Available tools:
- TOOL: ask(query) — Query your study documents. Use for:
  • Questions about specific topics ("what is X?", "explain Y")
  • Requests for explanations or definitions
  • Follow-ups and clarifications ("tell me more about that")

- TOOL: quiz(topic, count=5) — Generate practice questions. Use when:
  • User asks for a quiz or practice questions
  • User says "test me", "quiz me", "practice"

- TOOL: summarize(topic) — Create a topic summary. Use when:
  • User asks for a summary, overview, or key points
  • User says "summarize X", "give me the key concepts"

- TOOL: chat(text) — Casual conversation or general chat. Use when:
  • User greets you ("hi", "hello", "hey")
  • User gives thanks or feedback ("thanks", "good", "nice")
  • User asks about you or what you can do
  • The message is purely conversational with no study intent

Rules:
1. Always respond conversationally first, then add the TOOL: at the END.
2. For greetings/chat → use chat() tool with a friendly reply.
3. For "what is X" or "explain Y" → use ask().
4. For "quiz me" or "test me" → use quiz().
5. For "summarize X" or "key points" → use summarize().
6. For "tell me more" or "explain that" → use ask() with the follow-up question.
7. If user says a topic is too hard/easy → use ask() or quiz() with adjusted difficulty.
8. You can use MULTIPLE tools in one response if the request is compound.
9. Keep responses concise and friendly. Use emojis naturally.
10. Chat in the same language as the user (Indonesian or English).

Format for TOOL lines (put at the END of your response):
TOOL: tool_name(param1="value1", param2=123)

Examples:
User: "hello!"
You: Hey! Ready to study? 📖
TOOL: chat(text="Hey! I can help you study your documents. Ask me anything!")

User: "what is behavioral finance?"
You: Great question! Let me look that up in your documents...
TOOL: ask(query="What is behavioral finance?")

User: "quiz me on chapter 3"
You: Let me generate some practice questions for you! 🧠
TOOL: quiz(topic="chapter 3", count=5)

User: "explain loss aversion and quiz me on it"
You: Let me find information about loss aversion and then test your knowledge! 📚
TOOL: ask(query="Explain loss aversion in behavioral finance")
TOOL: quiz(topic="loss aversion", count=3)
"""

# ── TOOL regex ────────────────────────────────────────────
# Matches: TOOL: tool_name(param1="value1", param2=123)
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
        # Simple param parser: key="value" or key=number
        for part in raw_params.split(","):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                params[key] = val

    return name, params


# ── Tool implementations ──────────────────────────────────
async def _tool_ask(query: str) -> str:
    """Execute the ask() tool — RAG query."""
    from app.plugins.study.handler import ask_query
    try:
        return await ask_query(query)
    except Exception as e:
        logger.error("ask() failed: %s", e)
        return f"⚠️ Sorry, I couldn't find an answer: {e}"


async def _tool_quiz(topic: str, count: str = "5") -> str:
    """Execute the quiz() tool — generate practice questions."""
    from app.plugins.study.handler import generate_quiz
    try:
        n = max(1, min(20, int(count)))
    except (ValueError, TypeError):
        n = 5
    try:
        return await generate_quiz(topic, n)
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


async def _tool_chat(text: str) -> str:
    """Execute the chat() tool — direct Ollama conversation (no RAG)."""
    from app.rag.models import get_llm
    try:
        llm = get_llm()
        personality = settings.ai_personality.strip()
        prompt = f"Personality: {personality}\n\nRespond conversationally to: {text}" if personality else f"Respond conversationally to: {text}"
        response = await llm.acomplete(prompt)
        return str(response).strip()
    except Exception as e:
        logger.error("chat() failed: %s", e)
        return f"Hey! 😊 I'm here to help you study. Ask me anything about your documents!"


# ── Tool registry ─────────────────────────────────────────
_TOOL_REGISTRY = {
    "ask": _tool_ask,
    "quiz": _tool_quiz,
    "summarize": _tool_summarize,
    "chat": _tool_chat,
}


# ── BrainPlugin ───────────────────────────────────────────
class BrainPlugin(Plugin):
    name = "brain"
    commands = []  # No slash commands — intercepts all free text
    description = "AI agent that understands your intent and routes to the right tool"

    async def handle(self, ctx: BotContext) -> str | None:
        """Handle free text — send to Gemini agent, process tool calls."""
        message = ctx.message_text.strip()

        # ── 1. Call Gemini with agent prompt ──────────────
        logger.info("Brain: sending to Gemini: %s", message[:80])

        try:
            response = await gemini_chat(
                system_prompt=AGENT_PROMPT,
                user_message=message,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout,
            )
        except RuntimeError as e:
            logger.error("Gemini call failed: %s", e)
            # Fallback: try direct Ollama chat
            return await _tool_chat(
                f"The user said: {message}. Respond conversationally."
            )

        logger.info("Brain: Gemini raw response (first 100 chars): %s", response[:100])

        # ── 2. Check for TOOL: lines ─────────────────────
        if "TOOL:" not in response:
            return response  # No tools — return direct

        # ── 3. Process TOOL: lines ────────────────────────
        lines = response.split("\n")
        result_lines = []

        for line in lines:
            if "TOOL:" in line:
                parsed = _parse_tool(line)
                if parsed:
                    tool_name, params = parsed
                    tool_fn = _TOOL_REGISTRY.get(tool_name)
                    if tool_fn:
                        logger.info("Brain: executing tool '%s' with %s", tool_name, params)
                        tool_result = await tool_fn(**params)
                        result_lines.append(tool_result)
                    else:
                        logger.warning("Brain: unknown tool '%s'", tool_name)
                        result_lines.append(f"⚠️ Unknown tool: {tool_name}")
                else:
                    # Keep the line as-is if we can't parse it
                    result_lines.append(line)
            else:
                result_lines.append(line)

        return "\n\n".join(result_lines).strip()
