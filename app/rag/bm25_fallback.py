"""BM25 fallback search — lexical keyword search when vector search fails.

Used when cross-encoder reranker returns 0 relevant chunks.
Purely local, no API calls, ~100ms for 1000 chunks.
"""

import logging
import re
import time

logger = logging.getLogger(__name__)


def bm25_search(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Search chunks using BM25 keyword matching.

    Args:
        query: The search query.
        chunks: Full list of chunks from the index.
            Each dict must have 'text' key. 'filename' and 'score' are optional.
        top_k: Number of results to return.

    Returns:
        Top-k chunks ranked by BM25 score, with 'bm25_score' key added.
        Returns empty list if rank_bm25 not installed or no chunks provided.
    """
    if not chunks:
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank_bm25 not installed — skipping BM25 fallback")
        return []

    try:
        t0 = time.time()

        # Tokenize: simple whitespace + lowercase + strip punctuation
        def tokenize(text: str) -> list[str]:
            return re.findall(r"\w+", text.lower())

        corpus = [tokenize(c["text"]) for c in chunks]

        # Skip if corpus is empty after tokenization
        if not any(corpus):
            return []

        bm25 = BM25Okapi(corpus)

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        # Attach BM25 scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["bm25_score"] = float(score)

        # Sort by BM25 score descending
        ranked = sorted(chunks, key=lambda c: c.get("bm25_score", 0), reverse=True)

        # Filter: only chunks with positive BM25 score
        filtered = [c for c in ranked if c.get("bm25_score", 0) > 0]

        elapsed_ms = (time.time() - t0) * 1000
        logger.info(
            "BM25: %d chunks → %d relevant (%.0fms)",
            len(chunks),
            len(filtered),
            elapsed_ms,
        )

        return filtered[:top_k]

    except Exception as e:
        logger.error("BM25 search failed: %s", e)
        return []
