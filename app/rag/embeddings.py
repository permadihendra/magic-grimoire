"""Fallback embedding model using sentence-transformers.

If Ollama's embed endpoint fails (e.g. Go runner crash),
this provides a CPU-based fallback using all-MiniLM-L6-v2 (~22MB).
"""

import logging
from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)

_DEFAULT_BATCH_SIZE = DEFAULT_EMBED_BATCH_SIZE  # 10
_hf_model: BaseEmbedding | None = None


def _get_hf_embed_model() -> BaseEmbedding:
    """Load the HuggingFace sentence-transformers model (singleton)."""
    global _hf_model
    if _hf_model is not None:
        return _hf_model

    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info("Loading sentence-transformers fallback embed model: all-MiniLM-L6-v2")
        _hf_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embed_batch_size=32,
            device="cpu",
        )
        logger.info("sentence-transformers fallback embed model loaded")
        return _hf_model
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: uv pip install sentence-transformers"
        )


def _try_primary_call(primary: BaseEmbedding | None, method_name: str, *args: Any, **kwargs: Any) -> Any:
    """Try calling a method on the primary model, return None on failure."""
    if primary is None:
        return None
    try:
        method = getattr(primary, method_name)
        return method(*args, **kwargs)
    except Exception as e:
        logger.warning(
            "Ollama embed (%s) failed: %s — will use fallback if needed",
            method_name, e,
        )
        return None


class FallbackEmbedding(BaseEmbedding):
    """Embedding model that tries Ollama first, falls back to sentence-transformers.

    Usage: pass this as Settings.embed_model instead of OllamaEmbedding directly.

    Auto-switches to CPU fallback if Ollama's embed endpoint crashes
    (e.g. "signal arrived during cgo execution" = Go runner panic).
    """

    def __init__(self, primary: BaseEmbedding | None = None) -> None:
        super().__init__(
            embed_batch_size=getattr(primary, "embed_batch_size", _DEFAULT_BATCH_SIZE)
            if primary
            else _DEFAULT_BATCH_SIZE,
        )
        self._primary = primary
        self._fallback: BaseEmbedding | None = None
        self._using_fallback = False

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    @property
    def primary_model_name(self) -> str:
        if self._primary:
            return getattr(self._primary, "model_name", "unknown")
        return "none"

    @property
    def fallback_model_name(self) -> str:
        return "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def active_model_name(self) -> str:
        if self._using_fallback:
            return self.fallback_model_name
        return self.primary_model_name

    def _ensure_fallback(self) -> BaseEmbedding:
        """Load fallback model if not yet loaded. Marks _using_fallback=True."""
        if self._fallback is None:
            self._fallback = _get_hf_embed_model()
            self._using_fallback = True
            logger.warning(
                "Ollama embed failed — switched to CPU fallback: all-MiniLM-L6-v2. "
                "Embedding will be slower but indexing will succeed."
            )
        return self._fallback

    # ── Abstract method implementations ─────────────────────

    # Text embedding (used during indexing)
    async def _async_get_text_embedding(self, text: str) -> list[float]:
        if not self._using_fallback and self._primary is not None:
            try:
                return await self._primary._async_get_text_embedding(text)
            except Exception as e:
                logger.warning("Ollama text embed failed: %s — using fallback", e)
                self._ensure_fallback()
        return await self._fallback._async_get_text_embedding(text)

    async def _async_get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False
    ) -> list[list[float]]:
        if not self._using_fallback and self._primary is not None:
            try:
                return await self._primary._async_get_text_embedding_batch(
                    texts, show_progress=show_progress
                )
            except Exception as e:
                logger.warning(
                    "Ollama text embed batch (%d) failed: %s — using fallback",
                    len(texts), e,
                )
                self._ensure_fallback()
        return await self._fallback._async_get_text_embedding_batch(
            texts, show_progress=show_progress
        )

    def _get_text_embedding(self, text: str) -> list[float]:
        if not self._using_fallback and self._primary is not None:
            try:
                return self._primary._get_text_embedding(text)
            except Exception as e:
                logger.warning("Ollama text embed (sync) failed: %s — using fallback", e)
                self._ensure_fallback()
        return self._fallback._get_text_embedding(text)

    def _get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False
    ) -> list[list[float]]:
        if not self._using_fallback and self._primary is not None:
            try:
                return self._primary._get_text_embedding_batch(
                    texts, show_progress=show_progress
                )
            except Exception as e:
                logger.warning(
                    "Ollama text embed batch (sync, %d) failed: %s — using fallback",
                    len(texts), e,
                )
                self._ensure_fallback()
        return self._fallback._get_text_embedding_batch(
            texts, show_progress=show_progress
        )

    # Query embedding (used during retrieval)
    async def _aget_query_embedding(self, query: str) -> list[float]:
        if not self._using_fallback and self._primary is not None:
            try:
                return await self._primary._aget_query_embedding(query)
            except Exception as e:
                logger.warning("Ollama query embed async failed: %s — using fallback", e)
                self._ensure_fallback()
        return await self._fallback._aget_query_embedding(query)

    def _get_query_embedding(self, query: str) -> list[float]:
        if not self._using_fallback and self._primary is not None:
            try:
                return self._primary._get_query_embedding(query)
            except Exception as e:
                logger.warning("Ollama query embed (sync) failed: %s — using fallback", e)
                self._ensure_fallback()
        return self._fallback._get_query_embedding(query)