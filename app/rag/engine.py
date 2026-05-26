"""RAG query engine — retrieve relevant chunks and generate answers.

Uses the VectorStoreIndex to find relevant document chunks, then
sends them as context to the Ollama LLM for answer generation.
"""

import logging
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer

from app.config import settings
from app.rag.prompts import (
    QA_PROMPT,
    QUIZ_PROMPT,
    SUMMARY_PROMPT,
    get_study_prompt,
)

logger = logging.getLogger(__name__)


class RAGEngine:
    """High-level RAG query interface.

    Wraps LlamaIndex's query engine with study-specific prompts.
    """

    def __init__(self, index: VectorStoreIndex | None = None) -> None:
        self._index = index
        self._query_engine: RetrieverQueryEngine | None = None

    def set_index(self, index: VectorStoreIndex) -> None:
        """Set or update the underlying index."""
        self._index = index
        self._query_engine = None  # Reset engine, will be rebuilt

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

    async def query(self, question: str, mode: str = "qa") -> str:
        """Query the index with a question and return the answer.

        Args:
            question: The user's question or prompt.
            mode: 'qa' (default Q&A), 'quiz' (generate questions), 'summary'

        Returns:
            Generated response text.
        """
        engine = self._get_query_engine()

        # Craft a study-focused prompt with context
        if mode == "quiz":
            prompt_template = QUIZ_PROMPT
        elif mode == "summary":
            prompt_template = SUMMARY_PROMPT
        else:
            prompt_template = QA_PROMPT

        # Prepend the study prompt to guide the LLM
        full_prompt = f"{prompt_template}\n\nQuestion: {question}"
        personality = settings.ai_personality.strip()
        if personality:
            full_prompt = (
                f"Personality: {personality}\n\n{full_prompt}"
            )

        response = await engine.aquery(full_prompt)
        return str(response)

    async def query_with_sources(self, question: str) -> dict[str, Any]:
        """Query and return both answer and source citations.

        Returns:
            Dict with 'response' (str) and 'sources' (list of metadata).
        """
        engine = self._get_query_engine()
        full_prompt = get_study_prompt(question)

        response = await engine.aquery(full_prompt)

        sources = []
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("file_name", "Unknown"),
                "score": float(node.score) if node.score else 0.0,
                "text_preview": node.text[:200] if node.text else "",
            })

        return {
            "response": str(response),
            "sources": sources,
        }
