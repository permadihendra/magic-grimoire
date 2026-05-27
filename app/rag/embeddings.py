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
            "Run: uv add sentence-transformers"
        )


class FallbackEmbedding(BaseEmbedding):
    """Embedding model that tries Ollama first, falls back to sentence-transformers.

    Usage: pass this as Settings.embed_model instead of OllamaEmbedding directly.
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

    def _get_fallback(self) -> BaseEmbedding:
        if self._fallback is None:
            self._fallback = _get_hf_embed_model()
            self._using_fallback = True
            logger.warning(
                "Ollama embed endpoint failed — switched to CPU fallback: all-MiniLM-L6-v2. "
                "Embedding will be slower but indexing will succeed."
            )
        return self._fallback

    def _get_active(self) -> BaseEmbedding:
        if self._using_fallback:
            return self._fallback or self._get_fallback()
        if self._primary is not None:
            return self._primary
        return self._get_fallback()

    async def _async_get_text_embedding(self, text: str) -> list[float]:
        if not self._using_fallback and self._primary is not None:
            try:
                return await self._primary._async_get_text_embedding(text)
            except Exception as e:
                logger.warning("Ollama embed failed: %s — switching to fallback", e)
                self._get_fallback()
        return await self._get_fallback()._async_get_text_embedding(text)

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
                    "Ollama embed batch failed (%d texts): %s — switching to fallback",
                    len(texts), e,
                )
                self._get_fallback()
        return await self._get_fallback()._async_get_text_embedding_batch(
            texts, show_progress=show_progress
        )

    # Sync versions also go through async
    def _get_text_embedding(self, text: str) -> list[float]:
        # Use the sync path of the active model
        active = self._get_active()
        try:
            result = active.get_text_embedding(text)
            if self._using_fallback:
                return result
            # Try primary
            if self._primary:
                return self._primary.get_text_embedding(text)
            return result
        except Exception as e:
            if not self._using_fallback:
                logger.warning("Ollama embed failed (sync): %s — switching to fallback", e)
                self._get_fallback()
            return self._get_fallback().get_text_embedding(text)

    def _get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False
    ) -> list[list[float]]:
        if not self._using_fallback and self._primary is not None:
            try:
                return self._primary.get_text_embedding_batch(
                    texts, show_progress=show_progress
                )
            except Exception as e:
                logger.warning(
                    "Ollama embed batch failed (sync, %d texts): %s — switching to fallback",
                    len(texts), e,
                )
                self._get_fallback()
        return self._get_fallback().get_text_embedding_batch(
            texts, show_progress=show_progress
        )