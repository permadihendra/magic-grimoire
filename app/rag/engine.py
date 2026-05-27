"""RAG query engine — retrieve relevant chunks and generate answers.

Uses the VectorStoreIndex to find relevant document chunks, then
sends them as context to the Ollama LLM for answer generation.
Supports difficulty levels and source citations.
"""

import logging
import re
from typing import Any


def shorten_filename(filename: str) -> str:
    """Transform ugly filenames into clean, readable book titles.

    Examples:
      "Statman, Meir - Finance for normal people _ how investors ... (2017).pdf"
      → "Finance for Normal People"

      "Practical Laravel Develop clean MVC web applications (2022).pdf"
      → "Practical Laravel"
    """
    # Remove extension
    name = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Remove Zone.Identifier suffix
    name = name.replace(":Zone.Identifier", "")

    # Remove author prefix: "Author Name - " or "Author Name:"
    # Only match if the text before " - " looks like an author (contains a comma)
    # e.g., "Statman, Meir - Finance..." or "Kahneman, Daniel - Thinking..."
    if "," in name.split("-")[0]:
        name = re.sub(r"^[^\-]+,\s*[^\-]+\s*[-–]\s*", "", name)

    # Remove year in parentheses
    name = re.sub(r"\s*\(\d{4}\)\s*", "", name)

    # Remove publisher suffixes like "- Oxford University Press"
    name = re.sub(r"\s*[-–]\s*[A-Z][a-z]+\s+(University|Press|Books|Publishing|Inc|Ltd|House).*$", "", name, flags=re.IGNORECASE)

    # Remove subtitle after separators
    name = re.sub(r"\s*[:;]\s*.*$", "", name)

    # Remove trailing descriptor (everything after the main title idea)
    # e.g., "Practical Laravel Develop clean MVC web applications" → "Practical Laravel"
    # Keep first 4 meaningful words max
    name = re.sub(r"\s*[-–]\s*\d+(st|nd|rd|th)?\s*edition", "", name, flags=re.IGNORECASE)

    # Truncate subtitle BEFORE underscore replacement
    # Subtitles are often separated by " _ ", " – ", " - "
    for sep in [" _ ", " – ", " - "]:
        if sep in name:
            name = name.split(sep)[0]
            break

    # Underscores → spaces, then collapse
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    # Remove trailing year-like numbers
    name = re.sub(r"\s+\d{4}\s*$", "", name)

    # If still long (>4 words), strip subtitle/descriptor parts
    words = name.split()
    if len(words) > 4:
        # Truncation strategies:
        # 1. Split at lowercase-starting word (grammatical subtitle)
        # 2. Split at common verb/descriptor words (PascalCase descriptors)
        # 3. Split at position 4
        truncate_at = 4
        subtitle_verbs = {"develop", "building", "creating", "mastering", "learning",
                         "practical", "guide", "handbook", "introduction", "advanced",
                         "modern", "complete", "essential", "professional",
                         "clean", "mvc", "web", "applications"}
        for i in range(2, min(len(words), 10)):
            w = words[i]
            w_lower = w.lower().strip(",.;:!?")
            # Lowercase start = grammatical subtitle
            if w[0].islower() and w_lower not in {"a", "an", "the", "and", "or", "but",
                                                    "for", "nor", "yet", "so", "of",
                                                    "in", "on", "at", "to", "by", "with", "from"}:
                truncate_at = i
                break
            # Common subtitle/descriptor word (case-insensitive)
            if w_lower in subtitle_verbs:
                truncate_at = i
                break
        name = " ".join(words[:truncate_at])

    # Final safety truncation
    words = name.split()
    if len(words) > 7:
        name = " ".join(words[:6]) + "..."

    # Title case
    if len(name) > 3:
        name = name.title()

    return name if name else filename[:40]

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer

from app.config import settings
from app.rag.prompts import (
    get_qa_prompt,
    get_quiz_prompt,
    SUMMARY_PROMPT,

)

logger = logging.getLogger(__name__)

_MAX_SOURCES = 3  # max source citations to show


class RAGEngine:
    """High-level RAG query interface.

    Wraps LlamaIndex's query engine with study-specific prompts.
    Supports difficulty levels and returns source citations.
    """

    def __init__(self, index: VectorStoreIndex | None = None) -> None:
        self._index = index
        self._query_engine: RetrieverQueryEngine | None = None

    def set_index(self, index: VectorStoreIndex) -> None:
        """Set or update the underlying index."""
        self._index = index
        self._query_engine = None

    def _get_query_engine(self) -> RetrieverQueryEngine:
        """Get or create the query engine with current index."""
        if self._query_engine is not None:
            return self._query_engine

        if self._index is None:
            raise RuntimeError("No index available — index some documents first")

        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=settings.retrieval_top_k,
        )

        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            use_async=True,
        )

        self._query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return self._query_engine

    async def query(
        self,
        question: str,
        mode: str = "qa",
        difficulty: str = "normal",
        count: int = 5,
    ) -> str:
        """Query the index and return the answer.

        Args:
            question: The user's question or prompt.
            mode: 'qa', 'quiz', or 'summary'.
            difficulty: 'simple', 'normal', or 'advanced'.
            count: Number of quiz questions (quiz mode only).

        Returns:
            Generated response text.
        """
        engine = self._get_query_engine()

        if mode == "quiz":
            prompt_template = get_quiz_prompt(count, difficulty)
        elif mode == "summary":
            prompt_template = SUMMARY_PROMPT
        else:
            prompt_template = get_qa_prompt(difficulty)

        full_prompt = f"{prompt_template}\n\nQuestion: {question}"

        response = await engine.aquery(full_prompt)
        return str(response)

    async def query_with_sources(
        self,
        question: str,
        difficulty: str = "normal",
    ) -> dict[str, Any]:
        """Query and return BOTH answer and source citations.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of dicts).
        """
        engine = self._get_query_engine()
        prompt = get_qa_prompt(difficulty)
        full_prompt = f"{prompt}\n\nQuestion: {question}"

        response = await engine.aquery(full_prompt)

        # Extract unique sources with scores
        seen_files = set()
        sources = []
        for node in response.source_nodes:
            fname = node.metadata.get("file_name", "Unknown")
            if fname not in seen_files and len(sources) < _MAX_SOURCES:
                seen_files.add(fname)
                sources.append({
                    "filename": fname,
                    "score": float(node.score) if node.score else 0.0,
                })

        # Build answer with source citations appended
        answer = str(response)

        # Check confidence — if top source score is low, warn
        top_score = max((s["score"] for s in sources), default=0.0)
        low_confidence = top_score < 0.7

        # Deduplicate sources for the citation block
        cited = []
        seen_cited = set()
        for s in sources:
            if s["filename"] not in seen_cited:
                seen_cited.add(s["filename"])
                cited.append(s)

        if cited:
            citations = []
            seen = set()
            for s in cited:
                short = shorten_filename(s["filename"])
                if short not in seen:
                    seen.add(short)
                    # Add confidence indicator
                    if s["score"] < 0.7:
                        citations.append(f"📖 *Source:* {short} ⚠️ low relevance")
                    else:
                        citations.append(f"📖 *Source:* {short}")
            answer += "\n\n" + "\n".join(citations)

        # Append low-confidence warning if ALL sources are weak
        if low_confidence and cited:
            answer += (
                "\n\n⚠️ *Low confidence* — your documents may not cover "
                "this topic well. Try a different question or upload "
                "relevant material."
            )

        return {
            "answer": answer,
            "sources": sources,
        }
