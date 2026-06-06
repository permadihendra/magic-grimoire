"""Dispatcher — routes all incoming messages to the right handler.

Handles:
- Document uploads → confirm → save → index
- Slash commands → plugin
- Free text → BrainPlugin (Gemini agent)
"""

import logging
import os

from app.config import settings
from app.plugins.base import PluginRegistry, DispatchResult

logger = logging.getLogger(__name__)


# ── Pending file confirmations ───────────────────────────
# {chat_id: {"file_id": ..., "file_name": ..., "file_size": ...}}
_pending_confirm: dict[int, dict] = {}


# ── Telegram helpers ─────────────────────────────────────


async def _send_message(chat_id: int, text: str) -> dict | None:
    """Send a Telegram message and return result with message_id."""
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


async def _download_telegram_file(file_id: str, dest_path: str) -> bool:
    """Download a file from Telegram servers to a local path."""
    import httpx

    token = settings.telegram_token

    # Step 1: Get file path from Telegram
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"https://api.telegram.org/bot{token}/getFile",
            json={"file_id": file_id},
        )
        if resp.status_code != 200:
            logger.error("getFile failed: %s", resp.text)
            return False
        data = resp.json()
        if not data.get("ok") or "result" not in data:
            logger.error("getFile returned not ok: %s", data)
            return False
        file_path = data["result"]["file_path"]

    # Step 2: Download file
    download_url = f"https://api.telegram.org/file/bot{token}/{file_path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(download_url)
        if resp.status_code != 200:
            logger.error("File download failed: %s", resp.status_code)
            return False

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(resp.content)

    logger.info("Downloaded file %s (%d bytes)", dest_path, len(resp.content))
    return True


# ── File upload handler ──────────────────────────────────


async def _handle_document(message) -> None:
    """Handle a document/file upload — ask user for confirmation."""
    doc = message.document
    if not doc:
        return

    chat_id = message.chat_id
    file_id = doc.file_id
    file_name = doc.file_name or f"document_{file_id[:8]}"
    file_size = doc.file_size or 0

    # Validate file type — only documents we can index
    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".doc", ".epub", ".rtf", ".csv"}
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in allowed_extensions:
        await _send_message(
            chat_id,
            f"⚠️ Sorry, I can only index these file types:\n"
            f"`{', '.join(sorted(allowed_extensions))}`\n\n"
            f"Your file `{file_name}` was ignored.",
        )
        return

    # Size check (Telegram allows up to 50MB)
    size_mb = file_size / (1024 * 1024)
    if file_size > 50 * 1024 * 1024:
        await _send_message(
            chat_id,
            f"⚠️ File too large ({size_mb:.1f} MB). Maximum is 50 MB.",
        )
        return

    # Ask for confirmation
    _pending_confirm[chat_id] = {
        "file_id": file_id,
        "file_name": file_name,
        "file_size": file_size,
    }

    await _send_message(
        chat_id,
        f"📄 *Received:* `{file_name}` ({size_mb:.1f} MB)\n\n"
        f"Save this to your study documents? (reply `yes` or `no`)",
    )


async def _handle_confirmation(chat_id: int, text: str) -> DispatchResult | None:
    """Handle a yes/no confirmation for a pending file save."""
    pending = _pending_confirm.get(chat_id)
    if not pending:
        return None  # Not a confirmation — let normal flow handle it

    answer = text.lower().strip()
    if answer in ("yes", "y", "ya", "yeah"):
        file_id = pending["file_id"]
        file_name = pending["file_name"]
        file_size = pending["file_size"]

        safe_name = "".join(c for c in file_name if c.isalnum() or c in "._- ").strip()
        if not safe_name:
            safe_name = f"document_{file_id[:8]}.pdf"

        docs_dir = settings.docs_dir
        os.makedirs(docs_dir, exist_ok=True)
        dest_path = os.path.join(docs_dir, safe_name)

        await _send_message(chat_id, "⬇️ Downloading file...")

        success = await _download_telegram_file(file_id, dest_path)
        del _pending_confirm[chat_id]

        if success:
            size_mb = file_size / (1024 * 1024)
            reply = (
                f"✅ *File saved!*\n\n"
                f"📄 `{safe_name}` ({size_mb:.1f} MB) → `{docs_dir}/`\n\n"
                f"Now run `/index` to index this document, then you can ask me anything about it!"
            )
        else:
            reply = "⚠️ Failed to download file. Please try again or upload manually."

        return DispatchResult(reply=reply)

    if answer in ("no", "n", "nope", "cancel"):
        del _pending_confirm[chat_id]
        return DispatchResult(reply="🗑️ File ignored. You can upload again if you change your mind.")

    # Unclear response — stay in pending state
    await _send_message(
        chat_id,
        f"Reply `yes` to save `{pending['file_name']}` or `no` to cancel.",
    )
    return None


# ── Main dispatcher ──────────────────────────────────────


async def dispatch(update, thinking_msg_id: int | None = None) -> DispatchResult:
    """Route incoming messages to the right handler."""
    message = update.message or update.edited_message
    if not message:
        return DispatchResult(reply=None)

    chat_id = message.chat_id

    # ── 1. Document uploads ─────────────────────────────
    if message.document:
        await _handle_document(message)
        return DispatchResult(reply=None)

    # ── 2. Text messages ────────────────────────────────
    if not message.text:
        logger.debug("Received non-text update: %s", update.update_id)
        return DispatchResult(reply=None)

    text = message.text.strip()

    # ── 3. Confirmation responses (yes/no for pending saves) ──
    if text.lower() in ("yes", "y", "ya", "yeah", "no", "n", "nope", "cancel"):
        confirm = await _handle_confirmation(chat_id, text)
        if confirm:
            return confirm

    # Handle Telegram reply feature
    reply_to = message.reply_to_message
    if reply_to and reply_to.text:
        reply_preview = reply_to.text[:200]
        reply_user = reply_to.from_user.first_name if reply_to.from_user else "User"
        text = f'[Replying to {reply_user}: "{reply_preview}"]\n{text}'

    # Build context
    from app.bot.context import build_context
    ctx = build_context(update, text, thinking_msg_id=thinking_msg_id)

    # ── 4. Slash commands → plugin ──────────────────────
    if text.startswith("/"):
        command = text.split()[0].split("@")[0].lower().lstrip("/")
        registry = PluginRegistry.get()
        plugin = registry.resolve(command)
        if plugin:
            try:
                result = await plugin.handle(ctx)
                if result is None:
                    return DispatchResult(reply=None)
                if isinstance(result, DispatchResult):
                    return result
                return DispatchResult(reply=result)
            except Exception as e:
                logger.error("Plugin '%s' failed: %s", plugin.name, e, exc_info=True)
                error_detail = str(e)[:200]
                return DispatchResult(reply=f"⚠️ Error processing `/{command}`:\n`{error_detail}`")
        return DispatchResult(reply=None)

    # ── 5. Free text → BrainPlugin (Gemini agent) ───────
    registry = PluginRegistry.get()
    brain = registry.get_plugin("brain")
    if brain:
        try:
            result = await brain.handle(ctx)
            if result is None:
                return DispatchResult(reply=None)
            if isinstance(result, DispatchResult):
                return result
            return DispatchResult(reply=result)
        except Exception as e:
            logger.error("Brain plugin failed: %s", e, exc_info=True)
            study = registry.get_plugin("study")
            if study:
                try:
                    result = await study.handle(ctx)
                    if isinstance(result, DispatchResult):
                        return result
                    return DispatchResult(reply=result)
                except Exception as e2:
                    logger.error("Study fallback also failed: %s", e2)
    return DispatchResult(reply=None)