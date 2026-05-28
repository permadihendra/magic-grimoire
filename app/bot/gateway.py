import asyncio
import hmac
import json
import logging

from fastapi import APIRouter, Request, Response
from telegram import Update

from app.bot.dispatcher import dispatch
from app.bot.middlewares import check_rate_limit
from app.config import settings
from app.ui.helpers import _split_into_chunks

logger = logging.getLogger(__name__)

router = APIRouter()

# Track recently processed update_ids to ignore Telegram retries
_processed_updates: set[int] = set()
MAX_PROCESSED = 100  # keep last 100 IDs to avoid unbounded memory


async def _send_telegram_message(chat_id: int, text: str) -> dict | None:
    """Send a message via Telegram Bot API using raw httpx.
    Falls back to plain text if markdown causes 400 error.
    Returns response JSON (contains message_id) or None on failure.
    """
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
        else:
            logger.error("Send message failed: %s", resp.text)
            return None


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

    # Handle messages — text, document, or other
    message = update.message or update.edited_message
    if not message:
        logger.debug("Received update without message: %s", update.update_id)
        return Response(status_code=200)

    # Allow document messages (they have no text but we handle them)
    has_text = bool(message.text)
    has_doc = bool(message.document)

    if not has_text and not has_doc:
        logger.debug("Received non-text, non-document update: %s", update.update_id)
        return Response(status_code=200)

    log_text = message.text[:100] if has_text else f"[document: {message.document.file_name}]"
    logger.info("Update %s from chat %s: %s", update.update_id, message.chat_id, log_text)

    if not check_rate_limit(message.chat_id):
        await _send_telegram_message(
            message.chat_id, "⏱ Slow down! You're sending too many requests."
        )
        return Response(status_code=200)

    # Process — plugins return string or DispatchResult
    try:
        reply = await asyncio.wait_for(dispatch(update, None), timeout=300)
    except asyncio.TimeoutError:
        logger.error("Dispatch timed out after 300s")
        reply = "⏳ Processing timed out after 5 minutes.\nTry a simpler or more specific question."
    except Exception as e:
        logger.error("Dispatch failed: %s", e, exc_info=True)
        reply = "⚠️ Sorry, something went wrong processing your request."

    # Extract string from result (may be str or DispatchResult)
    from app.plugins.base import DispatchResult
    if isinstance(reply, DispatchResult):
        reply = reply.reply
    if not isinstance(reply, str) or not reply:
        # Silent commands (like /index with inline edits) return None or empty
        return Response(status_code=200)

    # Split and send — handles Telegram's 4096 char limit
    # For short messages, _split_into_chunks returns a single chunk
    chunks = _split_into_chunks(reply)
    for i, chunk in enumerate(chunks):
        await _send_telegram_message(message.chat_id, chunk)
        if i < len(chunks) - 1:
            await asyncio.sleep(0.3)

    return Response(status_code=200)