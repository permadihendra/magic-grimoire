from app.plugins.base import BotContext


def build_context(update, message_text: str, thinking_msg_id: int | None = None) -> BotContext:
    """Build a BotContext from a Telegram Update."""
    message = update.message or update.edited_message
    user = message.from_user if message else None

    return BotContext(
        chat_id=message.chat_id if message else 0,
        user_id=user.id if user else 0,
        username=user.username if user else None,
        message_text=message_text,
        is_group=message.chat.type in ("group", "supergroup") if message else False,
        raw_update=update,
        thinking_msg_id=thinking_msg_id,
    )
