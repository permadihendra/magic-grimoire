"""Query refiner — transform raw user questions into optimal search queries.

Uses Gemini to strip conversational filler and generate keyword variations.
Lightweight (~50ms), always runs before retrieval when enabled.
"""

import logging
import re
import time

logger = logging.getLogger(__name__)

_REFINER_SYSTEM = """You are a search query optimizer. Given a user question, produce 3-5 search queries optimized for vector similarity search.

Rules:
1. Strip conversational filler ("Can you tell me...", "I want to know...", "What does the book say about...")
2. Extract core intent and key terms
3. Generate 3-5 variations with different phrasings
4. Each variation should be 3-8 words
5. Include synonyms and related terms
6. Return ONLY the queries, one per line, no numbering or formatting

Example:
Input: "Can you tell me what the book says about loss aversion in finance?"
Output:
loss aversion finance
loss aversion prospect theory
loss aversion behavioral economics
psychology of financial losses
risk aversion investment decisions"""


async def refine_query(question: str) -> list[str]:
    """Refine a user question into multiple search query variations.

    Args:
        question: Raw user question.

    Returns:
        List of 3-5 refined query strings. Original question is always first.
    """
    from app.config import settings

    if not settings.use_query_refiner:
        return [question]

    try:
        from app.llm.gemini import gemini_chat

        t0 = time.time()
        response = await gemini_chat(
            system_prompt=_REFINER_SYSTEM,
            user_message=question,
            max_tokens=150,
            timeout=20.0,
        )
        elapsed_ms = (time.time() - t0) * 1000

        # Parse: split by newlines, clean up
        queries = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering prefixes: "1. ", "1) ", "- "
            line = re.sub(r"^[\d\-\*]+[\.\)]\s*", "", line)
            line = line.strip('"').strip("'").strip("`")
            if line and len(line) >= 3:
                queries.append(line)

        if not queries:
            logger.warning("Query refiner returned empty — using original")
            return [question]

        # Always include original question as first query
        if question not in queries:
            queries.insert(0, question)

        logger.info(
            "Query refiner: '%s' → %d queries (%.0fms)",
            question[:40],
            len(queries),
            elapsed_ms,
        )
        return queries[:5]  # Max 5 variations

    except Exception as e:
        logger.warning("Query refiner failed: %s — using original", e)
        return [question]
