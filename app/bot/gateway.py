import hmac
import json
import logging

from fastapi import APIRouter, Request, Response
from telegram import Update

from app.bot.dispatcher import dispatch
from app.bot.middlewares import check_rate_limit
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Track recently processed update_ids to ignore Telegram retries
_processed_updates: set[int] = set()
MAX_PROCESSED = 100  # keep last 100 IDs to avoid unbounded memory


async def _send_telegram_message(chat_id: int, text: str, keyboard=None) -> dict | None:
    """Send a message via Telegram Bot API using raw httpx.
    Falls back to plain text if markdown causes 400 error.
    Returns response JSON (contains message_id) or None on failure.
    """
    import httpx

    url = f"https://api.telegram.org/bot{settings.telegram_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if keyboard:
        payload["reply_markup"] = json.loads(keyboard) if isinstance(keyboard, str) else keyboard

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 400:
            logger.warning("Markdown send failed, retrying as plain text")
            payload.pop("parse_mode", None)
            resp = await client.post(url, json=payload)
        if resp.status_code == 200:
            return resp.json().get("result")
        else:
            logger.error("Send message failed: %s", resp.text)
            return None


async def _edit_message_text(chat_id: int, message_id: int, text: str) -> bool:
    """Edit a message's text. Returns True on success, False on failure."""
    import httpx

    url = f"https://api.telegram.org/bot{settings.telegram_token}/editMessageText"
    payload = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 400:
                logger.warning("Markdown edit failed, retrying as plain text")
                payload.pop("parse_mode", None)
                resp = await client.post(url, json=payload)
            return resp.status_code == 200
    except Exception as e:
        logger.warning("Failed to edit message %s: %s", message_id, e)
        return False


@router.post("/webhook")
async def webhook(request: Request) -> Response:
    """Telegram webhook receiver."""
    body = await request.body()

    if settings.telegram_webhook_secret:
        received_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not received_token:
            logger.warning("Webhook request missing secret token")
            return Response(status_code=401)
        if not hmac.compare_digest(received_token, settings.telegram_webhook_secret):
            logger.warning("Webhook secret token mismatch")
            return Response(status_code=403)

    try:
        data = json.loads(body)
        update = Update.de_json(data, None)
    except Exception as e:
        logger.error("Failed to parse Telegram update: %s", e)
        return Response(status_code=200)

    if not update:
        return Response(status_code=200)

    # ── Retry dedup: skip already-processed updates ──────────
    global _processed_updates
    if update.update_id in _processed_updates:
        logger.debug("Skipping duplicate update %s", update.update_id)
        return Response(status_code=200)
    _processed_updates.add(update.update_id)
    if len(_processed_updates) > MAX_PROCESSED:
        _processed_updates = set(list(_processed_updates)[-MAX_PROCESSED:])

    # Handle text messages only
    message = update.message or update.edited_message
    if not message or not message.text:
        logger.debug("Received non-text update: %s", update.update_id)
        return Response(status_code=200)

    logger.info(
        "Update %s from chat %s: %s",
        update.update_id,
        message.chat_id,
        message.text[:100],
    )

    if not check_rate_limit(message.chat_id):
        await _send_telegram_message(
            message.chat_id, "⏱ Slow down! You're sending too many requests."
        )
        return Response(status_code=200)

    # Send thinking indicator first, then process
    thinking_msg = await _send_telegram_message(
        message.chat_id, "⏳ Searching the grimoire…"
    )
    thinking_msg_id = thinking_msg.get("message_id") if thinking_msg else None

    try:
        reply = await dispatch(update, thinking_msg_id)
    except Exception as e:
        logger.error("Dispatch failed: %s", e, exc_info=True)
        reply = "⚠️ Sorry, something went wrong processing your request."

    if reply:
        if thinking_msg_id:
            ok = await _edit_message_text(message.chat_id, thinking_msg_id, reply)
            if not ok:
                await _send_telegram_message(message.chat_id, reply)
        else:
            await _send_telegram_message(message.chat_id, reply)
    else:
        if thinking_msg_id:
            await _edit_message_text(
                message.chat_id,
                thinking_msg_id,
                "🤔 Hmm, I couldn't find anything relevant. Try a different question or `/help`.",
            )

    return Response(status_code=200)


@router.get("/health")
async def health():
    return {"status": "ok", "model": settings.ollama_llm_model}
