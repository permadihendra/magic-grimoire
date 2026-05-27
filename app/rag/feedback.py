"""Feedback Learner — simple per-session relevance feedback.

When a user says "wrong source" or "that's from the wrong book",
the FeedbackLearner applies a penalty to that document for the session.
When they say "perfect", it applies a boost.

This is a contextual bandit approach: one action (prefer/avoid document),
immediate reward (user feedback), simple policy (score adjustment).
No model training needed.

Session state is cleared when the bot restarts.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Per-chat learners — keyed by Telegram chat_id
# Cleared on bot restart (acceptable — study sessions are short)
_learners: dict[int, "FeedbackLearner"] = {}


class FeedbackLearner:
    """Tracks per-document penalties and boosts for one chat session."""

    MAX_HISTORY = 10
    # Words that typically indicate partial document names in feedback
    INDICATOR_WORDS = {"from", "in", "the", "book", "document", "source", 
                       "wrong", "right", "correct", "incorrect", "not"}

    def __init__(self, chat_id: int):
        self.chat_id = chat_id
        self.penalized_docs: dict[str, float] = {}  # filename keyword → factor
        self.boosted_docs: dict[str, float] = {}    # filename keyword → factor
        self.history: list[dict] = []                # recent feedback events

    # ── Public API ───────────────────────────────────

    def penalize(self, doc_ref: str, factor: float = 0.5) -> None:
        """Apply a penalty to a document. factor=0.5 means 50% score reduction."""
        key = self._normalize_key(doc_ref)
        self.penalized_docs[key] = factor
        self.history.append({"type": "penalty", "doc": key, "factor": factor})
        self._trim_history()
        logger.info("FeedbackLearner[chat=%s]: penalized '%s' (x%s)", 
                     self.chat_id, key, factor)

    def boost(self, doc_ref: str, factor: float = 1.5) -> None:
        """Apply a boost to a document. factor=1.5 means 50% score increase."""
        key = self._normalize_key(doc_ref)
        self.boosted_docs[key] = factor
        self.history.append({"type": "boost", "doc": key, "factor": factor})
        self._trim_history()
        logger.info("FeedbackLearner[chat=%s]: boosted '%s' (x%s)", 
                     self.chat_id, key, factor)

    def adjust_score(self, filename: str, base_score: float) -> float:
        """Apply all active penalties and boosts to a retrieval score.

        Args:
            filename: The document filename from node metadata.
            base_score: The original retrieval score.

        Returns:
            Adjusted score.
        """
        score = base_score
        fname_lower = filename.lower()
        fname_normalized = self._normalize_key(fname_lower)

        # Apply penalties (reduce score)
        for key, factor in self.penalized_docs.items():
            if key in fname_normalized or self._partial_match(key, fname_normalized):
                score *= factor
                logger.debug("Feedback: penalty on '%s': %.2f → %.2f", 
                             filename, base_score, score)

        # Apply boosts (increase score)
        for key, factor in self.boosted_docs.items():
            if key in fname_normalized or self._partial_match(key, fname_normalized):
                score *= factor
                logger.debug("Feedback: boost on '%s': %.2f → %.2f", 
                             filename, base_score, score)

        return score

    def get_blocked_docs(self) -> set[str]:
        """Get documents that should be EXCLUDED entirely (penalty < 0.1)."""
        return {k for k, v in self.penalized_docs.items() if v < 0.2}

    def get_summary(self) -> str:
        """Human-readable summary of current feedback state."""
        parts = []
        if self.penalized_docs:
            parts.append(f"Penalized: {', '.join(self.penalized_docs)}")
        if self.boosted_docs:
            parts.append(f"Boosted: {', '.join(self.boosted_docs)}")
        return " · ".join(parts) if parts else "No feedback yet"

    def clear(self) -> None:
        """Reset all feedback for this session."""
        self.penalized_docs.clear()
        self.boosted_docs.clear()
        self.history.clear()

    # ── Internal helpers ─────────────────────────────

    @staticmethod
    def _normalize_key(s: str) -> str:
        """Normalize a document reference for matching."""
        s = s.lower()
        s = s.replace("_", " ").replace(".", " ").replace(",", " ")
        s = re.sub(r"\s+", " ", s).strip()
        # Remove common file extensions
        s = re.sub(r"\s*(\.pdf|\.epub|\.mobi|\.txt|\.docx)\s*$", "", s)
        return s

    @staticmethod
    def _partial_match(key: str, filename: str) -> bool:
        """Check if key words appear in filename (word-level match)."""
        key_words = set(key.split()) - FeedbackLearner.INDICATOR_WORDS
        fname_words = set(filename.split())
        if not key_words:
            return False
        return len(key_words & fname_words) >= len(key_words) * 0.5

    def _trim_history(self) -> None:
        if len(self.history) > self.MAX_HISTORY:
            self.history = self.history[-self.MAX_HISTORY:]


# ── Module-level access ──────────────────────────────

def get_learner(chat_id: int) -> FeedbackLearner:
    """Get or create a FeedbackLearner for a chat session."""
    if chat_id not in _learners:
        _learners[chat_id] = FeedbackLearner(chat_id)
    return _learners[chat_id]


def clear_learner(chat_id: int) -> None:
    """Remove a learner (e.g., on /start or session reset)."""
    _learners.pop(chat_id, None)


def all_learners() -> dict[int, FeedbackLearner]:
    """Get all active learners (for debugging)."""
    return _learners
