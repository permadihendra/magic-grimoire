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


def _build_status_footer(processing: dict | None = None) -> str:
    """Build a processing status footer for long-running / resource-heavy operations.

    Appended to query, quiz, and summarize responses.
    Shows backend (GPU/CPU/Cache), chunk count, and elapsed time.
    """
    if not processing:
        return ""

    backend = processing.get("backend", "")
    chunks = processing.get("chunks", 0)
    cache_hit = processing.get("cache_hit", False)
    elapsed_ms = processing.get("elapsed_ms", 0)

    if cache_hit:
        elapsed_s = elapsed_ms / 1000
        return f"\n\n🟢 *Cache hit* · {elapsed_s:.1f}s"

    # Backend emoji
    backend_map = {
        "gpu": ("🟢", "GPU"),
        "cpu": ("🟡", "CPU"),
        "fallback": ("🟡", "CPU"),
        "unknown": ("⚪", "—"),
        "cache": ("🟢", "Cache"),
    }
    emoji, label = backend_map.get(backend, ("⚪", backend or "—"))

    elapsed_s = elapsed_ms / 1000
    if chunks > 0:
        return f"\n\n{emoji} *{label}* · {chunks} chunks · {elapsed_s:.1f}s"
    return f"\n\n{emoji} *{label}* · {elapsed_s:.1f}s"


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
    logger.info(
        "Update %s from chat %s: %s",
        update.update_id,
        message.chat_id,
        log_text,
    )

    if not check_rate_limit(message.chat_id):
        await _send_telegram_message(
            message.chat_id, "⏱ Slow down! You're sending too many requests."
        )
        return Response(status_code=200)

    # Process — BrainPlugin manages thinking indicators internally for slow ops
    from app.plugins.base import DispatchResult
    try:
        import asyncio
        result = await asyncio.wait_for(dispatch(update, None), timeout=300)
    except asyncio.TimeoutError:
        logger.error("Dispatch timed out after 300s")
        result = DispatchResult(reply="⏳ Processing timed out after 5 minutes.\n"
                               "Try a simpler or more specific question.")
    except Exception as e:
        logger.error("Dispatch failed: %s", e, exc_info=True)
        result = DispatchResult(reply="⚠️ Sorry, something went wrong processing your request.")

    reply = result.reply if isinstance(result, DispatchResult) else (result if isinstance(result, str) else None)
    if not reply:
        return Response(status_code=200)
    if isinstance(result, DispatchResult) and result.processing:
        reply = reply + _build_status_footer(result.processing)
    if not isinstance(reply, str):
        logger.error("BUG: reply is not a string! type=%s value=%r", type(reply), reply)
        return Response(status_code=500)
    await _send_telegram_message(message.chat_id, reply)

    return Response(status_code=200)



