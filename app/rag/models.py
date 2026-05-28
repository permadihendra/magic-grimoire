"""Ollama model configuration for Magic Grimoire.

Manages the LLM and embedding model instances used by LlamaIndex.
All inference runs locally via Ollama — no cloud API costs.
"""

import asyncio
import logging

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from app.config import settings

logger = logging.getLogger(__name__)

_llm: Ollama | None = None
_embed_model: OllamaEmbedding | None = None
_warmed_up: bool = False


def get_llm() -> Ollama:
    """Get or create the Ollama LLM instance.

    Default: qwen3:8b — runs on 8GB GPU with ~5.2 GB VRAM.
    Timeout is generous (300s) to allow first-load model loading into VRAM.
    """
    global _llm
    if _llm is None:
        timeout = max(settings.ollama_timeout, 300.0)  # minimum 5 min
        logger.info(
            "Initializing Ollama LLM: %s at %s (timeout=%ss)",
            settings.ollama_llm_model,
            settings.ollama_base_url,
            timeout,
        )
        # Optimized for 8 GB VRAM (RTX 3050): Q4_0 quant + 4K context
        # temperature 0.0 = deterministic (also set in Modelfile — defense in depth)
        _llm = Ollama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            request_timeout=timeout,
            temperature=0.0,
            context_window=2048,
            additional_kwargs={
                "num_predict": 2048,   # actually passed to Ollama API (not silently dropped)
                "num_ctx": 4096,        # more headroom for long answers
            },
        )
    return _llm


def get_embed_model() -> OllamaEmbedding:
    """Get or create the Ollama embedding model.

    Default: nomic-embed-text — lightweight, good quality, ~0.5 GB VRAM.
    """
    global _embed_model
    if _embed_model is None:
        logger.info(
            "Initializing Ollama embed model: %s at %s",
            settings.ollama_embed_model,
            settings.ollama_base_url,
        )
        ollama_embed = OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
            embed_batch_size=3,          # Reduced: was 10 — prevents Go runner OOM during batch embed
            ollama_additional_kwargs={"timeout": settings.ollama_timeout},
        )
        # Wrap with fallback: if Ollama embed crashes (Go panic), use CPU sentence-transformers
        from app.rag.embeddings import FallbackEmbedding
        _embed_model = FallbackEmbedding(primary=ollama_embed)
    return _embed_model


def configure_settings() -> None:
    """Configure LlamaIndex global Settings with our Ollama models.

    Call once at startup before any indexing or querying.
    """
    from llama_index.core import Settings

    Settings.llm = get_llm()
    Settings.embed_model = get_embed_model()
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap
    logger.info(
        "LlamaIndex configured: llm=%s embed=%s chunk_size=%d",
        settings.ollama_llm_model,
        settings.ollama_embed_model,
        settings.chunk_size,
    )


async def warm_up(progress_callback=None) -> None:
    """Pre-load the LLM model into VRAM by sending a tiny chat request.

    This avoids the 30-60s cold-start delay on the first real query.
    Call once at startup after configure_settings().
    """
    global _warmed_up
    if _warmed_up:
        return

    llm = get_llm()
    logger.info("Warming up model %s (loading into VRAM)...", settings.ollama_llm_model)

    if progress_callback:
        await progress_callback("⚡ Warming up AI model...")

    try:
        # Send a minimal chat to force model loading
        import ollama
        from app.rag.guard import OllamaGuard

        async with OllamaGuard("model warm-up", require_health=False):
            client = ollama.AsyncClient(host=settings.ollama_base_url)
            # Wrap in timeout — guard's timeout param isn't auto-applied
            await asyncio.wait_for(
                client.chat(
                    model=settings.ollama_llm_model,
                    messages=[{"role": "user", "content": "Say OK"}],
                    options={"num_predict": 2},
                ),
                timeout=60.0,  # 60s max for warm-up
            )
        _warmed_up = True
        logger.info("Model %s warmed up successfully", settings.ollama_llm_model)
        if progress_callback:
            await progress_callback("✅ AI model ready")
    except Exception as e:
        logger.warning("Model warm-up failed (will load on first query): %s", e)
        if progress_callback:
            await progress_callback("⚠️ Model warm-up deferred to first query")


def reset_models() -> None:
    """Reset cached model instances (useful for testing/reload)."""
    global _llm, _embed_model, _warmed_up
    _llm = None
    _embed_model = None
    _warmed_up = False
