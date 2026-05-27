"""Knowledge Cache — Q&A pairs for fast repeat queries.

Stores previously answered questions with embeddings for cosine similarity
lookup. When a similar question is asked again, returns cached answer
instead of running full LlamaIndex retrieval.
"""

import asyncio
import logging
import struct
import time
from typing import Optional

import numpy as np

from app.config import settings
from app.database import get_db

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 10
_MIN_SIMILARITY = 0.85
_MAX_PAIRS = 500


async def search_cache(
    question: str,
    document: str | None = None,
    chat_id: int | None = None,
) -> dict | None:
    """Find a matching cached Q&A pair.

    Args:
        question: User's current question.
        document: Optional document name to match.
        chat_id: Chat ID for session-scoped search.

    Returns:
        Dict with question, full_answer, source_doc, similarity, or None.
    """
    try:
        question_embedding = await _embed_question(question)
        if question_embedding is None:
            return None

        db = await get_db()
        cursor = await db.execute(
            "SELECT id, question, question_embedding, full_answer, source_doc, "
            "retrieval_score, approved, used_count "
            "FROM knowledge_pairs "
            "ORDER BY created_at DESC LIMIT ?",
            (_MAX_CANDIDATES,),
        )
        rows = await cursor.fetchall()
        if not rows:
            return None

        best = None
        best_combined = 0.0

        for row in rows:
            if not row["question_embedding"]:
                continue
            stored_vec = _blob_to_vector(row["question_embedding"])
            query_sim = _cosine_similarity(question_embedding, stored_vec)

            # Source document matching
            source_sim = 1.0
            if document and row["source_doc"]:
                source_sim = _fuzzy_source_match(document, row["source_doc"])

            # Weighted combination
            combined = query_sim * 0.65 + source_sim * 0.35

            if combined > best_combined and combined >= _MIN_SIMILARITY:
                best_combined = combined
                best = {
                    "id": row["id"],
                    "question": row["question"],
                    "full_answer": row["full_answer"],
                    "source_doc": row["source_doc"],
                    "similarity": round(combined, 3),
                    "approved": bool(row["approved"]),
                }

        if best and best["similarity"] >= 0.92:
            await _increment_used(best["id"])
            logger.info(
                "Cache HIT: sim=%.3f (query=%.2f, source=%.2f) Q=%s",
                best["similarity"],
                query_sim,
                source_sim,
                question[:50],
            )
            return best

        if best and best["similarity"] >= _MIN_SIMILARITY:
            # Medium confidence — return with note
            await _increment_used(best["id"])
            best["full_answer"] += "\n\n💾 *Similar to a previous answer*"
            return best

    except Exception as e:
        logger.debug("Cache search failed: %s", e)

    return None


async def store_pair(
    question: str,
    answer: str,
    source_doc: str | None = None,
    retrieval_score: float = 0.0,
    chat_id: int | None = None,
    approved: bool = False,
) -> None:
    """Store a Q&A pair in the knowledge cache."""
    try:
        question_embedding = await _embed_question(question)
        blob = _vector_to_blob(question_embedding) if question_embedding is not None else None

        short_answer = answer[:500] if len(answer) > 500 else answer

        db = await get_db()
        await db.execute(
            "INSERT INTO knowledge_pairs "
            "(question, question_embedding, short_answer, full_answer, "
            "source_doc, retrieval_score, approved, chat_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                question, blob, short_answer, answer,
                source_doc, retrieval_score, int(approved), chat_id,
            ),
        )
        await db.commit()

        # Cleanup old entries
        await db.execute(
            "DELETE FROM knowledge_pairs WHERE id NOT IN "
            "(SELECT id FROM knowledge_pairs ORDER BY created_at DESC LIMIT ?)",
            (_MAX_PAIRS,),
        )
        await db.commit()

        logger.debug("Cache stored: %s (approved=%s)", question[:50], approved)
    except Exception as e:
        logger.debug("Cache store failed: %s", e)


async def invalidate_cache() -> None:
    """Clear all knowledge cache entries."""
    db = await get_db()
    await db.execute("DELETE FROM knowledge_pairs")
    await db.commit()
    logger.info("Knowledge cache invalidated")


# ── Internal helpers ──────────────────────────────────

async def _embed_question(text: str) -> np.ndarray | None:
    """Embed a question using nomic-embed-text."""
    try:
        from app.rag.models import get_embed_model
        embed_model = get_embed_model()
        embedding = embed_model.get_text_embedding(text)
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        logger.debug("Embed failed: %s", e)
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _fuzzy_source_match(doc_a: str, doc_b: str) -> float:
    """Fuzzy match two document names. Returns 0.0 to 1.0."""
    a = doc_a.lower().replace("_", " ").replace(".", " ")
    b = doc_b.lower().replace("_", " ").replace(".", " ")
    a_words = set(a.split())
    b_words = set(b.split())
    stop = {"a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "pdf", "epub"}
    a_words -= stop
    b_words -= stop
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / max(len(a_words), len(b_words))


def _vector_to_blob(vec: np.ndarray) -> bytes:
    """Convert numpy array to BLOB."""
    return struct.pack(f"{len(vec)}f", *vec)


def _blob_to_vector(blob: bytes) -> np.ndarray:
    """Convert BLOB to numpy array."""
    count = len(blob) // 4
    return np.array(struct.unpack(f"{count}f", blob), dtype=np.float32)


async def _increment_used(pair_id: int) -> None:
    """Increment the usage counter for a cached pair."""
    try:
        db = await get_db()
        await db.execute(
            "UPDATE knowledge_pairs SET used_count = used_count + 1, "
            "last_used_at = CURRENT_TIMESTAMP WHERE id = ?",
            (pair_id,),
        )
        await db.commit()
    except Exception:
        pass
