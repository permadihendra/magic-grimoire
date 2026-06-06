"""Cross-encoder reranker — re-score retrieved chunks for relevance.

Uses sentence-transformers CrossEncoder on GPU (fallback to CPU).
Model: cross-encoder/ms-marco-MiniLM-L6-v2 (~80MB, fast on GPU).

VRAM: ~210MB on RTX 3050 — fits within headroom alongside LLM + embed.
"""

import logging
import time

logger = logging.getLogger(__name__)

_model = None
_model_device: str | None = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker model.

    Tries GPU first, falls back to CPU on OOM.
    Model stays loaded between calls (no reload overhead).
    """
    global _model, _model_device
    if _model is not None:
        return _model, _model_device

    from app.config import settings

    device = settings.reranker_device

    # Try requested device first
    for attempt_device in [device, "cpu"]:
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker: %s on %s", settings.reranker_model, attempt_device)
            _model = CrossEncoder(
                settings.reranker_model,
                device=attempt_device,
                max_length=512,
            )
            _model_device = attempt_device
            logger.info("Reranker loaded on %s", attempt_device)
            return _model, _model_device
        except Exception as e:
            logger.warning("Reranker load failed on %s: %s", attempt_device, e)
            if attempt_device == "cpu":
                # Both attempts failed
                logger.error("Reranker unavailable — all devices failed")
                return None, None
            # Try CPU next
            continue

    return None, None


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Re-score chunks using cross-encoder and return top_k.

    Args:
        query: The search query.
        chunks: List of chunk dicts with 'text', 'filename', 'score' keys.
            May also contain '_node' key (original LlamaIndex node reference).
        top_k: Number of chunks to return (default: settings.reranker_top_k).

    Returns:
        Re-sorted list of chunks with added 'rerank_score' key.
        Returns original chunks[:top_k] if reranker unavailable.
    """
    from app.config import settings

    if top_k is None:
        top_k = settings.reranker_top_k

    if not chunks:
        return []

    model, device = _get_reranker()
    if model is None:
        logger.warning("Reranker unavailable — returning original chunks (top %d)", top_k)
        return chunks[:top_k]

    # Acquire global GPU lock — only one GPU operation at a time
    import asyncio
    from app.rag.guard import _gpu_lock
    try:
        _gpu_lock.acquire_nowait()
    except asyncio.LockedError:
        logger.info("Reranker: GPU busy — skipping rerank, returning original chunks")
        return chunks[:top_k]

    try:
        t0 = time.time()

        # Build (query, document) pairs for cross-encoder
        pairs = [(query, c["text"]) for c in chunks]

        # Predict relevance scores (logits, not probabilities)
        # Positive = relevant, negative = irrelevant
        scores = model.predict(pairs, show_progress_bar=False)

        # Attach rerank scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank score descending
        reranked = sorted(chunks, key=lambda c: c.get("rerank_score", 0), reverse=True)

        # Filter by threshold (keep chunks with score >= threshold)
        threshold = settings.reranker_threshold
        filtered = [c for c in reranked if c.get("rerank_score", 0) >= threshold]

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "Reranker: %d → %d chunks (threshold=%.2f, %.0fms, device=%s)",
            len(chunks),
            len(filtered),
            threshold,
            elapsed_ms,
            device,
        )

        return filtered[:top_k]

    except Exception as e:
        logger.error("Reranker failed: %s — returning original chunks", e)
        return chunks[:top_k]
    finally:
        _gpu_lock.release()


def rerank_available() -> bool:
    """Check if reranker model is loaded and available."""
    model, _ = _get_reranker()
    return model is not None


def reset_reranker() -> None:
    """Reset cached model (for testing)."""
    global _model, _model_device
    _model = None
    _model_device = None
