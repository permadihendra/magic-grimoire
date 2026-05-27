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


def _filename_match(fname: str, doc_ref: str) -> bool:
    """Check if a filename matches a document reference."""
    f = fname.lower().replace("_", " ").replace(".", " ")
    d = doc_ref.lower().replace("_", " ").replace(".", " ")
    f_words = set(f.split())
    d_words = set(d.split()) - {"a", "an", "the", "and", "or", "but", "for", "nor",
                                "of", "in", "on", "at", "to", "by", "with", "from",
                                "pdf", "epub", "book", "books"}
    if not d_words:
        return False
    return len(d_words & f_words) >= len(d_words) * 0.5


def _lookup_display_name(filename: str) -> str:
    """Get display name for a filename. Falls back to shorten_filename."""
    return shorten_filename(filename)


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

    def retrieve_only(self, question: str, document: str | None = None, chat_id: int | None = None) -> list[dict]:
        """Retrieve passages without LLM generation.

        Args:
            question: The query.
            document: Optional document name to boost scores for.
            chat_id: Telegram chat ID for feedback learning.

        Returns:
            List of dicts with keys: text, filename, score.
        """
        retriever = self._get_retriever()
        nodes = retriever.retrieve(question)

        # Apply document boosting
        if document:
            for node in nodes:
                fname = node.metadata.get("file_name", "")
                if _filename_match(fname, document):
                    node.score = (node.score or 0) * 2.0

        # Apply feedback learner adjustments
        if chat_id is not None:
            from app.rag.feedback import get_learner
            learner = get_learner(chat_id)
            for node in nodes:
                fname = node.metadata.get("file_name", "")
                node.score = learner.adjust_score(fname, node.score or 0)

        # Post-retrieval filtering
        if document:
            doc_nodes = []
            other_nodes = []
            for node in nodes:
                if _filename_match(node.metadata.get("file_name", ""), document):
                    doc_nodes.append(node)
                else:
                    other_nodes.append(node)
            if doc_nodes:
                doc_nodes.sort(key=lambda n: n.score or 0, reverse=True)
                other_nodes.sort(key=lambda n: n.score or 0, reverse=True)
                forced_count = min(2, len(doc_nodes))
                remaining = max(0, 5 - forced_count)
                nodes = doc_nodes[:forced_count] + other_nodes[:remaining]
        else:
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
        chat_id: int | None = None,
    ) -> dict[str, Any]:
        """Query and return BOTH answer and source citations.

        Args:
            question: The user's question.
            difficulty: 'simple', 'normal', or 'advanced'.
            document: Optional document name to boost scores for.
            chat_id: Telegram chat ID for feedback learning.

        Returns:
            Dict with 'answer' (str) and 'sources' (list of dicts).
        """
        # --- Phase 0: Knowledge cache check ---
        if chat_id is not None:
            try:
                from app.rag.knowledge_cache import search_cache
                cached = await search_cache(question, document, chat_id)
                if cached and cached["similarity"] >= 0.92:
                    return {
                        "answer": cached["full_answer"],
                        "sources": [{
                            "filename": cached.get("source_doc", "Unknown"),
                            "score": cached["similarity"],
                        }],
                    }
            except Exception as e:
                logger.debug("Cache check failed: %s", e)

        # --- Phase 1: Sync retrieval with document boosting + feedback ---
        retriever = self._get_retriever()
        nodes = retriever.retrieve(question)

        # Load feedback learner for this chat (if available)
        learner = None
        if chat_id is not None:
            from app.rag.feedback import get_learner
            learner = get_learner(chat_id)

        # Document-aware boosting (FTS5 resolves name → exact match)
        if document:
            for node in nodes:
                fname = node.metadata.get("file_name", "")
                if _filename_match(fname, document):
                    node.score = (node.score or 0) * 2.0  # 2x for named doc chunks

        # Apply feedback learner adjustments (PLAN retrieval quality Phase 2)
        blocked_docs = set()
        if learner:
            blocked_docs = learner.get_blocked_docs()
            for node in nodes:
                fname = node.metadata.get("file_name", "")
                node.score = learner.adjust_score(fname, node.score or 0)

        # Post-retrieval filtering: force named-doc chunks into context
        if document:
            doc_nodes = []
            other_nodes = []
            for node in nodes:
                fname = node.metadata.get("file_name", "")
                if _filename_match(fname, document):
                    doc_nodes.append(node)
                else:
                    other_nodes.append(node)

            if doc_nodes:
                # Group-sort: sort within each group, keep doc_nodes priority
                doc_nodes.sort(key=lambda n: n.score or 0, reverse=True)
                other_nodes.sort(key=lambda n: n.score or 0, reverse=True)
                forced_count = min(2, len(doc_nodes))
                remaining = max(0, 5 - forced_count)
                nodes = doc_nodes[:forced_count] + other_nodes[:remaining]
        else:
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
            for i, s in enumerate(cited):
                short = _lookup_display_name(s["filename"])
                if short not in seen:
                    seen.add(short)
                    label = "📖 *Primary source:*" if i == 0 else "📎 *Also:*"
                    if s["score"] < 0.7:
                        citations.append(f"{label} {short} ⚠️ low relevance")
                    else:
                        citations.append(f"{label} {short}")
            answer += "\n\n" + "\n".join(citations)

        if low_confidence and cited:
            answer += (
                "\n\n⚠️ *Low confidence* — your documents may not cover "
                "this topic well. Try a different question or upload "
                "relevant material."
            )

        # Store in knowledge cache for future queries
        if chat_id is not None and len(answer.strip()) >= 100:
            try:
                from app.rag.knowledge_cache import store_pair
                import asyncio as _asyncio
                _asyncio.create_task(
                    store_pair(question, answer, 
                               source_doc=sources[0]["filename"] if sources else None,
                               retrieval_score=top_score,
                               chat_id=chat_id, approved=False)
                )
            except Exception as e:
                logger.debug("Cache store failed: %s", e)

        return {
            "answer": answer,
            "sources": sources,
        }
