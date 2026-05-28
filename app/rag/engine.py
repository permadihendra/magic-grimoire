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
import time
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


def _backend() -> str:
    """Return the active embedding backend: 'gpu' (Ollama) or 'cpu' (sentence-transformers)."""
    try:
        from app.rag.embeddings import get_embed_backend
        return get_embed_backend()
    except Exception:
        return "unknown"


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
        chunk_texts = [n.text for n in nodes if n.text]
        total_tokens = sum(len(t.split()) * 1.3 for t in chunk_texts)
        logger.info("Retrieval(query): %d chunks, ~%d tokens", len(chunk_texts), int(total_tokens))

        if mode == "quiz":
            prompt_template = get_quiz_prompt(count, difficulty)
        elif mode == "summary":
            prompt_template = SUMMARY_PROMPT
        else:
            prompt_template = get_qa_prompt(difficulty)

        # For quiz/summary: use tree_summarize with the prompt as system instruction
        from app.rag.models import get_llm
        from app.rag.guard import OllamaGuard, OllamaBusyError, OllamaDeadError

        try:
            async with OllamaGuard("question answering", timeout=180):
                llm = get_llm()
                from llama_index.core.response_synthesizers import TreeSummarize
                synthesizer = TreeSummarize(llm=llm)
                response = await synthesizer.aget_response(
                    question,
                    chunk_texts,
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
        mode: str = "qa",  # "qa" | "quiz" | "summary" | "qna"
        count: int = 10,
        existing_pairs: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Query and return BOTH answer and source citations.

        Args:
            question: The user's question.
            difficulty: 'simple', 'normal', or 'advanced'.
            document: Optional document name to boost scores for.
            chat_id: Telegram chat ID for feedback learning.
            mode: 'qa', 'quiz', 'summary', or 'qna' (controls prompt template).
            count: Number of items for quiz/qna modes.
            existing_pairs: Previous Q&A pairs for complement-on-rerun.

        Returns:
            Dict with 'answer' (str), 'sources' (list), and 'processing' (dict).
        """
        _qid = f"q_{int(time.time()*1000)%100000:05d}"
        _t0 = time.time()
        logger.info("[%s] >>> q_with_sources START q='%s' diff=%s doc=%s",
                    _qid, question[:60], difficulty, document)

        # --- GUARDRAIL: Index health check ---
        if self._index:
            try:
                doc_count = len(self._index.docstore.docs)
                if doc_count == 0:
                    logger.warning("[%s] Index has 0 documents!", _qid)
                    return {
                        "answer": "📭 *Index is empty.* Run `/index` to rebuild it.",
                        "sources": [],
                        "processing": {"backend": "unknown", "chunks": 0, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000},
                    }
                logger.info("[%s]  index health: %d documents", _qid, doc_count)
            except Exception as e:
                logger.warning("[%s]  index health check failed: %s", _qid, e)

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
                        "processing": {"backend": "cache", "chunks": 0, "cache_hit": True, "elapsed_ms": (time.time()-_t0)*1000},
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
                "processing": {"backend": _backend(), "chunks": 0, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000},
            }

        # --- GUARDRAIL: Validate chunks before LLM call ---
        chunk_texts = [n.text for n in nodes if n.text]
        chunk_count = len(chunk_texts)

        if chunk_count == 0:
            logger.warning("[%s] No chunks retrieved — skipping LLM call", _qid)
            return {
                "answer": (
                    "📭 *No relevant passages found.*\n\n"
                    "Check that `ebooklib` + `html2text` are installed for EPUB support.\n"
                    "Run `/index` to rebuild the index."
                ),
                "sources": [],
                "processing": {"backend": _backend(), "chunks": 0, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000},
            }

        # Token budget check
        total_input_tokens = sum(len(t.split()) * 1.3 for t in chunk_texts)
        system_prompt_tokens = 250
        available_for_response = 2048 - total_input_tokens - system_prompt_tokens
        if total_input_tokens > 1800:
            logger.warning("[%s] Token budget high: ~%d input + %d system > 2048 ctx",
                          _qid, int(total_input_tokens), system_prompt_tokens)
        if available_for_response < 50:
            logger.error("[%s] Insufficient context headroom: %d tokens", _qid, int(available_for_response))
            return {
                "answer": "⚠️ *Context window too full.* Try a more specific question.",
                "sources": sources,
                "processing": {"backend": _backend(), "chunks": chunk_count, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000},
            }

        # --- Phase 2: LLM generation with timeout + fallback ---
        logger.info("[%s]  [p2] LLM START chunks=%d ~%d tokens (headroom=%d)",
                    _qid, chunk_count, int(total_input_tokens), int(available_for_response))

        from app.rag.models import get_llm
        from app.rag.guard import OllamaGuard, OllamaBusyError, OllamaDeadError

        # Select prompt template based on mode
        if mode == "quiz":
            from app.rag.prompts import get_quiz_prompt
            prompt_template = get_quiz_prompt(count=5, difficulty=difficulty)
            prompt_var = "query_str"
        elif mode == "summary":
            from app.rag.prompts import SUMMARY_PROMPT
            prompt_template = SUMMARY_PROMPT
            prompt_var = "query_str"
        elif mode == "qna":
            from app.rag.prompts import get_qna_prompt
            from llama_index.core.prompts import PromptTemplate
            prompt_template = PromptTemplate(
                get_qna_prompt(count=count, existing_pairs=existing_pairs)
            )
            prompt_var = "query_str"
        else:  # qa
            prompt_template = None
            prompt_var = None

        try:
            async with OllamaGuard("answer generation", timeout=180) as _guard:
                llm = get_llm()
                logger.info("[%s]  [p2] LLM instance: %s", _qid, getattr(llm, 'model', '?'))

                from llama_index.core.response_synthesizers import TreeSummarize
                kwargs = {"llm": llm, "use_async": True}
                if prompt_template is not None:
                    kwargs["summary_template"] = prompt_template
                synthesizer = TreeSummarize(**kwargs)
                logger.info("[%s]  [p2] TreeSummarize created, calling aget_response()...", _qid)

                _t2 = time.time()
                response = await asyncio.wait_for(
                    synthesizer.aget_response(query_str=question, text_chunks=chunk_texts),
                    timeout=120.0,  # 120s for synthesis (was 60s)
                )
                _gen_ms = (time.time()-_t2)*1000
                answer = str(response)
                logger.info("[%s]  [p2] LLM DONE %d chars (%.0fms gen, %.0fms total)",
                            _qid, len(answer), _gen_ms, (time.time()-_t0)*1000)

        except asyncio.TimeoutError:
            logger.error("[%s]  [p2] TreeSummarize timed out after 30s — fallback to simple prompt", _qid)
            # Fallback: simple join of top 2 chunks only
            fallback_context = "\n\n".join(chunk_texts[:2])
            fallback_prompt = (
                "Based ONLY on this context, answer the question.\n\n"
                f"Context: {fallback_context}\\n\\n"
                f"Question: {question}\\n\\n"
                f"Answer:"
            )
            try:
                async with OllamaGuard("answer generation fallback", timeout=60) as _guard:
                    llm = get_llm()
                    response = await asyncio.wait_for(llm.acomplete(fallback_prompt), timeout=60)
                    answer = str(response)
                    answer += "\n\n⚠️ *Response truncated due to timeout. Try a shorter question.*"
                    logger.info("[%s]  [p2] Fallback OK %d chars", _qid, len(answer))
            except Exception as e:
                logger.error("[%s]  [p2] Fallback also failed: %s", _qid, e)
                return {
                    "answer": "⚠️ *Query timed out. Try again with a shorter or more specific question.*",
                    "sources": sources,
                    "processing": {"backend": _backend(), "chunks": chunk_count, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000},
                }

        except OllamaBusyError as e:
            logger.error("[%s]  [p2] Ollama BUSY: %s", _qid, e)
            return {"answer": str(e), "sources": sources,
                   "processing": {"backend": _backend(), "chunks": chunk_count, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000}}
        except OllamaDeadError as e:
            logger.error("[%s]  [p2] Ollama DEAD: %s", _qid, e)
            return {"answer": "⚠️ Study engine unavailable. Try again in 30s.",
                   "sources": sources,
                   "processing": {"backend": _backend(), "chunks": chunk_count, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000}}
        except Exception as e:
            logger.error("[%s]  [p2] LLM CRASHED: %s", _qid, e, exc_info=True)
            raise

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
                "processing": {"backend": _backend(), "chunks": chunk_count, "cache_hit": False, "elapsed_ms": (time.time()-_t0)*1000},
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

        # --- Build follow-up (separate message) ---
        follow_up_parts = []

        # Next-step suggestion
        follow_up_parts.append(
            "💡 *Next steps:* Would you like a follow-up question, a `quiz` on this topic, "
            "or a `summary` of the key points? Just ask!"
        )

        # Diagnostics footer: chunk count + token size
        chunk_texts = [n.text for n in nodes if n.text]
        total_tokens = sum(len(t.split()) * 1.3 for t in chunk_texts)
        total_words = int(total_tokens * 0.75)  # ~0.75 words per token
        diag = (
            f"📊 *Retrieval:* {len(chunk_texts)} chunks | "
            f"~{int(total_tokens)} tokens (~{total_words} words) | "
            f"k={settings.retrieval_top_k} | "
            f"ctx=2048"
        )
        follow_up_parts.append(diag)

        if low_confidence and cited:
            follow_up_parts.append(
                "⚠️ *Low confidence* — your documents may not cover "
                "this topic well. Try a different question or upload "
                "relevant material."
            )

        follow_up = "\n\n".join(follow_up_parts)

        # Remove old next-step and diagnostics from answer (they were added earlier)
        # We rebuild the answer to only include answer + citations

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

        _total_ms = (time.time()-_t0)*1000
        logger.info("[%s] >>> q_with_sources DONE %d chars (%.0fms total)",
                    _qid, len(answer), _total_ms)

        return {
            "answer": answer,
            "sources": sources,
            "follow_up": follow_up,
            "processing": {"backend": _backend(), "chunks": chunk_count, "cache_hit": False, "elapsed_ms": _total_ms},
        }
