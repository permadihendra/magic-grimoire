"""Ollama Guardrails - prevent concurrent calls, detect crashes.

Single entry point for ALL Ollama operations. Ensures:
- Only ONE Ollama operation at a time (VRAM safety)
- Health check before every call
- Graceful error messages for users
- Crash detection and recovery
"""

import asyncio
import logging
import time
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Module state
_gpu_lock = asyncio.Semaphore(1)   # Global GPU lock — ONE operation at a time
_ollama_sem = _gpu_lock             # Alias: OllamaGuard uses same lock
_current_operation: str | None = None
_current_chat_id: int | None = None
_last_health: tuple[float, bool] = (0, True)
_health_cache_ttl = 5.0


class OllamaBusyError(Exception):
    """Another Ollama operation is already in progress."""


class OllamaDeadError(Exception):
    """Ollama service is not responding."""


# Health check

async def ollama_is_alive(timeout: float = 2.0) -> bool:
    """Check if Ollama is alive. Cached for 5 seconds."""
    global _last_health
    now = time.time()
    if now - _last_health[0] < _health_cache_ttl:
        return _last_health[1]

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            alive = r.status_code == 200
    except Exception:
        alive = False

    _last_health = (now, alive)
    if not alive:
        logger.warning("Ollama health check FAILED at %s", settings.ollama_base_url)
    return alive



async def ollama_check_vram() -> tuple[bool, str]:
    """Check if we have enough VRAM for inference. Returns (ok, message)."""
    try:
        import asyncio
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            return True, "VRAM check timed out"

        result = stdout.decode().strip()
        if result:
            used, total = result.split(",")
            used_mb = float(used.strip())
            total_mb = float(total.strip())
            free_mb = total_mb - used_mb
            if free_mb < 1800:
                return False, f"Low VRAM: {int(free_mb)}MB free (need ~2300MB)"
            return True, f"VRAM OK: {int(free_mb)}MB free"
    except Exception as e:
        logger.debug("VRAM check failed: %s", e)
    return True, "VRAM check unavailable"

def is_ollama_busy() -> bool:
    return _ollama_sem.locked()


def get_current_operation() -> str | None:
    return _current_operation


# Guarded execution

class OllamaGuard:
    """Context manager for guarded Ollama operations.

    Usage:
        try:
            async with OllamaGuard("ask", chat_id=123) as guard:
                result = await some_ollama_call()
        except OllamaBusyError as e:
            # Show friendly busy message
        except OllamaDeadError as e:
            # Show "Ollama down" message
    """

    def __init__(
        self,
        operation: str,
        chat_id: int | None = None,
        timeout: float = 120.0,
        require_health: bool = True,
    ):
        self.operation = operation
        self.chat_id = chat_id
        self.timeout = timeout
        self.require_health = require_health
        self._started: float | None = None

    async def __aenter__(self) -> "OllamaGuard":
        global _current_operation, _current_chat_id

        # 1. Health check
        if self.require_health:
            if not await ollama_is_alive():
                raise OllamaDeadError(
                    "Study engine unavailable.\n\n"
                    "Ollama isn't running. It should auto-restart.\n"
                    "Try again in 30 seconds."
                )

        # 2. VRAM pre-check
        vram_ok, vram_msg = await ollama_check_vram()
        logger.info("Ollama Guard VRAM: %s", vram_msg)

        # 3. Concurrency check
        if _ollama_sem.locked():
            raise OllamaBusyError(
                f"Already processing: {_current_operation or 'unknown'}.\n\n"
                "Please wait for it to finish, then try again."
            )

        # 4. Acquire lock
        try:
            await asyncio.wait_for(_ollama_sem.acquire(), timeout=5.0)
        except asyncio.TimeoutError:
            raise OllamaBusyError(
                "Another operation is finishing. Try again shortly."
            )

        _current_operation = self.operation
        _current_chat_id = self.chat_id
        self._started = time.time()

        logger.info(
            "Ollama Guard: lock acquired for '%s' (chat=%s)",
            self.operation, self.chat_id,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        global _current_operation, _current_chat_id
        elapsed = time.time() - (self._started or 0) if self._started else 0

        _current_operation = None
        _current_chat_id = None
        _ollama_sem.release()

        if exc_type:
            logger.error(
                "Ollama Guard: '%s' FAILED after %.1fs: %s",
                self.operation, elapsed, exc_val,
            )
            return False

        logger.info(
            "Ollama Guard: '%s' completed in %.1fs",
            self.operation, elapsed,
        )
        return False


# Startup verification

async def verify_ollama_on_startup() -> dict[str, Any]:
    """Verify Ollama and required models are available on startup."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            if r.status_code != 200:
                return {"ok": False, "models": [], "error": "Ollama not responding"}

            data = r.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            required = {settings.ollama_llm_model, settings.ollama_embed_model}
            missing = required - set(models)

            if missing:
                return {
                    "ok": False,
                    "models": models,
                    "error": f"Missing models: {', '.join(missing)}",
                }

            return {"ok": True, "models": models, "error": None}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


# Crash recovery

async def attempt_ollama_restart() -> bool:
    """Try to restart the Ollama service. Returns True if successful."""
    import subprocess
    try:
        subprocess.run(["pkill", "-f", "ollama serve"], timeout=5, capture_output=True)
        await asyncio.sleep(2)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.sleep(3)
        return await ollama_is_alive()
    except Exception as e:
        logger.error("Ollama restart failed: %s", e)
        return False
