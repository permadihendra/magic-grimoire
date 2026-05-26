"""RAG query engine — retrieve relevant chunks and generate answers.

Uses the VectorStoreIndex to find relevant document chunks, then
sends them as context to the Ollama LLM for answer generation.
Supports difficulty levels and source citations.
"""

import logging
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer

from app.config import settings
from app.rag.prompts import (
    get_qa_prompt,
    get_quiz_prompt,
    SUMMARY_PROMPT,
    DOC_SOURCE_FMT,
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

        # Deduplicate sources for the citation block
        cited = []
        seen_cited = set()
        for s in sources:
            if s["filename"] not in seen_cited:
                seen_cited.add(s["filename"])
                cited.append(s)

        if cited:
            citations = []
            for s in cited:
                citations.append(f"📖 *Source:* `{s['filename']}`")
            answer += "\n\n" + "\n".join(citations)

        return {
            "answer": answer,
            "sources": sources,
        }
