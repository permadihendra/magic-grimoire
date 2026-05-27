"""RAG query engine — retrieve relevant chunks and generate answers.

Supports:
- Document-aware retrieval (boost scores for named documents)
- Difficulty levels (simple/normal/advanced)
- Source citations with score indicators
- Retrieve-only mode (no LLM generation)
- Sync retrieval (async retrieval hangs in current LlamaIndex)
"""

import asyncio
import logging
import re
from typing import Any

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

from app.config import settings
from app.rag.prompts import (
    get_qa_prompt,
    get_quiz_prompt,
    SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)

_MAX_SOURCES = 3


def shorten_filename(filename: str) -> str:
    """Transform ugly filenames into clean, readable book titles."""
    name = filename.rsplit(".", 1)[0] if "." in filename else filename
    name = name.replace(":Zone.Identifier", "")
    name = re.sub(r"^[^\-]+,\s*[^\-]+\s*[-–]\s*", "", name)
    name = re.sub(r"\s*\(\d{4}\)\s*", "", name)
    name = re.sub(r"\s*[-–]\s*[A-Z][a-z]+\s+(University|Press|Books|Publishing|Inc|Ltd|House).*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*[:;]\s*.*$", "", name)
    name = re.sub(r"\s*[-–]\s*\d+(st|nd|rd|th)?\s*edition", "", name, flags=re.IGNORECASE)
    for sep in [" _ ", " – ", " - "]:
        if sep in name:
            name = name.split(sep)[0]
            break
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()
    name = re.sub(r"\s+\d{4}\s*$", "", name)
    words = name.split()
    if len(words) > 4:
        truncate_at = 4
        subtitle_verbs = {"develop", "building", "creating", "mastering", "learning",
                         "practical", "guide", "handbook", "introduction", "advanced",
                         "modern", "complete", "essential", "professional",
                         "clean", "mvc", "web", "applications"}
        for i in range(2, min(len(words), 10)):
            w = words[i]
            w_lower = w.lower().strip(",. ;:!?")
            if w[0].islower() and w_lower not in {"a", "an", "the", "and", "or", "but",
                                                    "for", "nor", "yet", "so", "of",
                                                    "in", "on", "at", "to", "by", "with", "from"}:
                truncate_at = i
                break
            if w_lower in subtitle_verbs:
                truncate_at = i
                break
        name = " ".join(words[:truncate_at])
    if len(name) > 3:
        name = name.title()
    return name if name else filename[:40]


class RAGEngine:
    """High-level RAG query interface."""

    def __init__(self, index: VectorStoreIndex | None = None) -> None:
        self._index = index
        self._retriever: VectorIndexRetriever | None = None

    def set_index(self, index: VectorStoreIndex) -> None:
        self._index = index
        self._retriever = None

    def _get_retriever(self) -> VectorIndexRetriever:
        if self._retriever is not None:
            return self._retriever
        if self._index is None:
            raise RuntimeError("No index available")
        self._retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=settings.retrieval_top_k,
        )
        return self._retriever

    # ── Retrieve only (no LLM generation) ─────────────────

    def retrieve_only(self, question: str, document: str | None = None) -> list[dict]:
        """Retrieve passages without LLM generation.

        Args:
            question: The query.
            document: Optional document name to boost scores for.

        Returns:
            List of dicts with keys: text, filename, score.
        """
        retriever = self._get_retriever()
        nodes = retriever.retrieve(question)

        # Apply document boosting (PLAN #6 + PLAN audit fix)
        if document:
            doc_words = set(document.lower().replace("_", " ").replace(".", " ").split())
            stopwords = {"a", "an", "the", "and", "or", "but", "for", "nor", "of",
                        "in", "on", "at", "to", "by", "with", "from", "pdf", "epub"}
            doc_words -= stopwords
            for node in nodes:
                fname = node.metadata.get("file_name", "").lower()
                fname_norm = fname.replace("_", " ").replace(".", " ")
                fname_words = set(fname_norm.split())
                if doc_words:
                    overlap = len(doc_words & fname_words) / len(doc_words)
                    if overlap >= 0.5:
                        boost = 1.0 + (overlap * 1.0)
                        node.score = (node.score or 0) * boost
            nodes.sort(key=lambda n: n.score or 0, reverse=True)

        passages = []
        seen_files = set()
        for node in nodes:
            fname = node.metadata.get("file_name", "Unknown")
            if fname not in seen_files or True:
                # Allow multiple from same doc
                passages.append({
                    "text": node.text or "",
                    "filename": fname,
                    "score": float(node.score) if node.score else 0.0,
                })
        return passages

    # ── Legacy query (no sources) ─────────────────────────

    async def query(
        self,
        question: str,
        mode: str = "qa",
        difficulty: str = "normal",
        count: int = 5,
    ) -> str:
        """Query the index and return the answer."""
        retriever = self._get_retriever()
        nodes = retriever.retrieve(question)
        context = "\n\n".join(n.text for n in nodes if n.text)

        if mode == "quiz":
            prompt_template = get_quiz_prompt(count, difficulty)
        elif mode == "summary":
            prompt_template = SUMMARY_PROMPT
        else:
            prompt_template = get_qa_prompt(difficulty)

        from app.rag.models import get_llm
        from app.rag.guard import OllamaGuard, OllamaBusyError, OllamaDeadError

        full_prompt = f"{prompt_template}\n\nContext:\n{context}\n\nQuestion: {question}"

        try:
            async with OllamaGuard("question answering", timeout=120):
                llm = get_llm()
                response = await asyncio.wait_for(
                    llm.acomplete(full_prompt), timeout=120
                )
                return str(response)
        except OllamaBusyError as e:
            return str(e)
        except OllamaDeadError:
            return "\u26a0\ufe0f Study engine unavailable. It should auto-restart. Try again in 30s."
        except asyncio.TimeoutError:
            return "\u26a0\ufe0f Query timed out. Try a simpler question."

    # ── Query with sources + document boosting ────────────

    async def query_with_sources(
        self,
        question: str,
        difficulty: str = "normal",
        document: str | None = None,
    ) -> dict[str, Any]:
        """Query and return BOTH answer and source citations.

        Args:
            question: The user's question.
            difficulty: 'simple', 'normal', or 'advanced'.
            document: Optional document name to boost scores for.
                      If user mentions a specific doc, pass it here.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of dicts).
        """
        # --- Phase 1: Sync retrieval with optional document boosting ---
        retriever = self._get_retriever()
        nodes = retriever.retrieve(question)

        # Document-aware boosting (PLAN #6 + PLAN audit fix: normalize names)
        if document:
            # Normalize: replace separators with spaces, collapse whitespace
            doc_words = set(document.lower().replace("_", " ").replace(".", " ").split())
            # Filter out common stopwords
            stopwords = {"a", "an", "the", "and", "or", "but", "for", "nor", "of",
                        "in", "on", "at", "to", "by", "with", "from", "pdf", "epub"}
            doc_words -= stopwords

            for node in nodes:
                fname = node.metadata.get("file_name", "").lower()
                fname_norm = fname.replace("_", " ").replace(".", " ")
                fname_words = set(fname_norm.split())
                # How many query words match the filename?
                if doc_words:
                    overlap = len(doc_words & fname_words) / len(doc_words)
                    if overlap >= 0.5:  # At least 50% of query words match
                        boost = 1.0 + (overlap * 1.0)  # 1.5x for 50%, 2.0x for 100%
                        node.score = (node.score or 0) * boost
            nodes.sort(key=lambda n: n.score or 0, reverse=True)

        # Extract unique sources
        seen_files = set()
        sources = []
        for node in nodes:
            fname = node.metadata.get("file_name", "Unknown")
            if fname not in seen_files and len(sources) < _MAX_SOURCES:
                seen_files.add(fname)
                sources.append({
                    "filename": fname,
                    "score": float(node.score) if node.score else 0.0,
                })

        # --- Validate: no source nodes found ---
        if not sources:
            return {
                "answer": (
                    "📭 *No relevant passages found in your documents.*\n\n"
                    "This could mean:\n"
                    "• The topic isn't covered in your documents\n"
                    "• The documents failed to parse correctly\n\n"
                    "Try:\n"
                    "• A different question or keywords\n"
                    "• `/index` to re-index your documents\n"
                    "• Uploading new material"
                ),
                "sources": [],
            }

        # --- Phase 2: Async LLM generation ---
        context = "\n\n".join(n.text for n in nodes if n.text)
        prompt = get_qa_prompt(difficulty)
        full_prompt = f"{prompt}\n\nContext:\n{context}\n\nQuestion: {question}"

        from app.rag.models import get_llm
        from app.rag.guard import OllamaGuard, OllamaBusyError, OllamaDeadError

        try:
            async with OllamaGuard("answer generation", timeout=120):
                llm = get_llm()
                response = await asyncio.wait_for(
                    llm.acomplete(full_prompt), timeout=120
                )
                answer = str(response)
        except OllamaBusyError as e:
            return {
                "answer": str(e),
                "sources": sources,
            }
        except OllamaDeadError:
            return {
                "answer": (
                    "\u26a0\ufe0f *Study engine unavailable*\n\n"
                    "Ollama isn't running. It should auto-restart shortly.\n"
                    "Try again in 30 seconds."
                ),
                "sources": sources,
            }
        except asyncio.TimeoutError:
            return {
                "answer": (
                    "\u26a0\ufe0f *Query timed out after 2 minutes*\n\n"
                    "Try:\n"
                    "\u2022 A simpler or shorter question\n"
                    "\u2022 Limiting to one document with `/files`\n"
                    "\u2022 Running `/index` to optimize the index"
                ),
                "sources": sources,
            }

        # --- Validate: answer is too short ---
        if len(answer.strip()) < 20:
            return {
                "answer": (
                    "📭 *I couldn't extract a clear answer from your documents.*\n\n"
                    "This might mean the relevant section wasn't parsed correctly.\n"
                    "Try `/index` to re-index, or check if LiteParse is installed:\n"
                    "`uv sync --extra liteparse`"
                ),
                "sources": sources,
            }

        # --- Phase 3: Build source citations ---
        top_score = max((s["score"] for s in sources), default=0.0)
        low_confidence = top_score < 0.7

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
                    if s["score"] < 0.7:
                        citations.append(f"📖 *Source:* {short} ⚠️ low relevance")
                    else:
                        citations.append(f"📖 *Source:* {short}")
            answer += "\n\n" + "\n".join(citations)

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
