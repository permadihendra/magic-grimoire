"""Progress watcher — phase-aware asyncio task for long operations.

Provides a ProgressState dataclass for shared state + a ProgressWatcher
asyncio task that reads the state every N seconds and sends intelligent
contextual updates to the user.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ProgressState:
    """Shared state between main task and watcher task.

    The main task writes to this as phases change.
    The watcher reads it periodically to build reports.
    """

    phase: str = "idle"  # retrieve | generate | index_parse | index_embed
    started_at: float = 0.0
    phase_started_at: float = 0.0

    # Phase-specific metrics (set by main task as data becomes available)
    passages_found: int = 0
    passages_docs: list[str] = field(default_factory=list)
    tokens_generated: int = 0
    words_generated: int = 0
    chunks_embedded: int = 0
    chunks_total: int = 0
    files_parsed: int = 0
    files_total: int = 0
    query_text: str = ""
    document_name: str | None = None

    # Status
    error: str | None = None
    complete: bool = False

    def start_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started_at = time.time()
        if self.started_at == 0.0:
            self.started_at = self.phase_started_at


class ProgressWatcher:
    """Asyncio task that watches ProgressState and sends Telegram updates.

    Usage:
        state = ProgressState()
        state.start_phase("generate")

        watcher = ProgressWatcher(chat_id, message_id, state, interval=60)
        watcher.start()

        # Main work happens here...
        state.tokens_generated = 120
        state.complete = True

        watcher.stop()
    """

    def __init__(
        self,
        chat_id: int,
        message_id: int,
        state: ProgressState,
        interval: int = 60,
    ):
        self.chat_id = chat_id
        self.message_id = message_id
        self.state = state
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._reports_sent: int = 0
        self._bot_token = settings.telegram_token

    def start(self) -> None:
        """Launch the watcher task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._watch_loop())
        logger.debug(
            "ProgressWatcher started for chat=%s, interval=%ds",
            self.chat_id, self.interval,
        )

    def stop(self) -> None:
        """Cancel the watcher task."""
        if self._task:
            self._task.cancel()
            self._task = None

    async def _edit(self, text: str) -> None:
        """Edit the watched message. Truncates if too long."""
        if len(text) > 4000:
            text = text[:3975] + "\n... (truncated)"
        try:
            url = f"https://api.telegram.org/bot{self._bot_token}/editMessageText"
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, json={
                    "chat_id": self.chat_id,
                    "message_id": self.message_id,
                    "text": text,
                    "parse_mode": None,  # Plain text to avoid Markdown issues
                })
                if r.status_code == 400:
                    # Retry without parse_mode (already None — try anyway)
                    pass
        except Exception as e:
            logger.debug("ProgressWatcher edit failed: %s", e)

    async def _watch_loop(self) -> None:
        """Send phase-aware updates with escalating frequency."""
        try:
            # Phase 0: Immediate report (t=0.5s — let the message settle first)
            await asyncio.sleep(0.5)
            if not self.state.complete:
                msg = self._build_initial_report()
                if msg:
                    await self._edit(msg)
                    self._reports_sent += 1

            # Phase 1: Escalating intervals — 30s, 30s, 15s, 15s, 15s...
            intervals = [30, 30, 15, 15, 15, 15, 15, 15]
            for interval in intervals:
                await asyncio.sleep(interval)
                if self.state.complete:
                    return
                msg = self._build_report()
                if msg:
                    await self._edit(msg)
                    self._reports_sent += 1
        except asyncio.CancelledError:
            pass

    def _build_initial_report(self) -> Optional[str]:
        """First report at t=0 — what's about to happen."""
        if self.state.phase == "generate":
            if self.state.passages_found > 0:
                short = ", ".join(
                    self._short_name(n) for n in (self.state.passages_docs or [])[:2]
                )
                return (
                    f"Found {self.state.passages_found} passages from {short}. "
                    "Generating answer... (this may take 10-60s)"
                )
            return "Generating answer from retrieved passages... (est. 10-60s)"
        elif self.state.phase in ("index_embed", "index_parse"):
            total = self.state.files_total or self.state.chunks_total or "?"
            return f"Starting indexing ({total} items)... (est. 30-90s)"
        return None

    def _build_report(self) -> Optional[str]:
        """Build a contextual progress report based on current phase."""
        elapsed = int(time.time() - self.state.phase_started_at)
        total_elapsed = int(time.time() - self.state.started_at)

        if self.state.phase == "generate":
            if self.state.words_generated > 0:
                # Streaming: we have word count
                return (
                    f"Writing answer: {self.state.words_generated} words so far "
                    f"({elapsed}s elapsed). Working..."
                )
            return self._escalating_message(elapsed, total_elapsed)

        elif self.state.phase == "index_embed":
            done = self.state.chunks_embedded
            total = self.state.chunks_total
            if total > 0 and done > 0:
                pct = min(done * 100 // total, 99)
                rate = done / elapsed if elapsed > 0 else 0
                remaining = ""
                if rate > 0 and done < total:
                    est = int((total - done) / rate)
                    remaining = f" ~{est}s remaining."
                return (
                    f"Embedding chunks: {done}/{total} ({pct}%)."
                    f"{remaining} ({elapsed}s elapsed)"
                )
            return f"Embedding chunks... ({elapsed}s elapsed)"

        elif self.state.phase == "index_parse":
            done = self.state.files_parsed
            total = self.state.files_total
            if total > 0:
                return (
                    f"Parsing documents: {done}/{total} files "
                    f"({elapsed}s elapsed)"
                )
            return f"Parsing documents... ({elapsed}s elapsed)"

        elif self.state.phase == "retrieve":
            if self.state.passages_found > 0:
                short = ", ".join(
                    self._short_name(n) for n in self.state.passages_docs[:2]
                )
                return (
                    f"Found {self.state.passages_found} passages from {short}. "
                    "Preparing answer..."
                )
            return None  # Too early to report

        # Generic fallback
        if elapsed < self.interval:
            return None
        return f"Still working... ({elapsed}s elapsed)"

    def _escalating_message(self, elapsed: int, total: int) -> Optional[str]:
        """Escalate tone as time increases."""
        if elapsed < 15:
            return None  # Too early to nag
        elif elapsed < 45:
            return (
                f"Still writing your answer ({elapsed}s). "
                "Working through the material."
            )
        elif elapsed < 90:
            return (
                f"Taking a bit longer ({elapsed}s). "
                "Generating a thorough response."
            )
        elif elapsed < 180:
            return (
                f"Complex question — still working ({elapsed}s). "
                "If this persists, try a simpler phrasing."
            )
        else:
            return (
                f"Very long generation ({elapsed}s). "
                "May time out. Try /index or a shorter question."
            )

    @staticmethod
    def _short_name(filename: str) -> str:
        """Quick abbreviation for display."""
        name = filename.rsplit(".", 1)[0] if "." in filename else filename
        name = name.replace("_", " ")
        if len(name) > 30:
            name = name[:27] + "..."
        return name
