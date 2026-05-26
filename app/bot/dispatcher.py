import logging

from app.plugins.base import PluginRegistry

logger = logging.getLogger(__name__)


async def dispatch(update, thinking_msg_id: int | None = None) -> str | None:
    """Route incoming messages to the right handler.

    Flow:
    1. Slash commands → resolved to plugin directly
    2. Free text → BrainPlugin (Gemini agent) → routes to StudyPlugin tools
    """
    message = update.message or update.edited_message
    if not message or not message.text:
        return None

    text = message.text.strip()
    chat_id = message.chat_id

    # Handle Telegram reply feature — inject replied text as context
    reply_to = message.reply_to_message
    if reply_to and reply_to.text:
        reply_preview = reply_to.text[:200]
        reply_user = reply_to.from_user.first_name if reply_to.from_user else "User"
        text = f'[Replying to {reply_user}: "{reply_preview}"]\n{text}'

    # Build context with thinking message id
    from app.bot.context import build_context
    ctx = build_context(update, text, thinking_msg_id=thinking_msg_id)

    # 1. Slash commands → route to plugin
    if text.startswith("/"):
        command = text.split()[0].split("@")[0].lower().lstrip("/")
        registry = PluginRegistry.get()
        plugin = registry.resolve(command)
        if plugin:
            try:
                return await plugin.handle(ctx)
            except Exception as e:
                logger.error("Plugin '%s' failed: %s", plugin.name, e, exc_info=True)
                return f"⚠️ Error processing `/{command}`."
        return None

    # 2. Free text → BrainPlugin (Gemini agent)
    registry = PluginRegistry.get()
    brain = registry.get_plugin("brain")
    if brain:
        try:
            reply = await brain.handle(ctx)
            return reply
        except Exception as e:
            logger.error("Brain plugin failed: %s", e, exc_info=True)
            # Fallback: try StudyPlugin directly
            study = registry.get_plugin("study")
            if study:
                try:
                    return await study.handle(ctx)
                except Exception as e2:
                    logger.error("Study fallback also failed: %s", e2)
    return None
