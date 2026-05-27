import sys, re

path = "/home/hendra/my-projects/magic-grimoire/app/rag/engine.py"
with open(path) as f:
    data = f.read()

# ── P0: Add entry log + qid ─────────────────────────────────
old_p0 = """        # --- Phase 0: Knowledge cache check ---
        if chat_id is not None:
            try:
                from app.rag.knowledge_cache import search_cache
                cached = await search_cache(question, document, chat_id)
                if cached and cached[\"similarity\"] >= 0.92:
                    return {
                        \"answer\": cached[\"full_answer\"],
                        \"sources\": [{
                            \"filename\": cached.get(\"source_doc\", \"Unknown\"),
                            \"score\": cached[\"similarity\"],
                        }],
                    }
            except Exception as e:
                logger.debug(\"Cache check failed: %s\", e)

        # --- Phase 1"""

new_p0 = """        _qid = f\"q_{int(time.time()*1000)%100000:05d}\"
        logger.info(\"[%s] >>> q_with_sources START q='%s' diff=%s doc=%s\",
                    _qid, question[:60], difficulty, document)
        _t0 = time.time()

        # --- Phase 0: Knowledge cache check ---
        if chat_id is not None:
            try:
                from app.rag.knowledge_cache import search_cache
                logger.info(\"[%s]  [p0] cache check...\", _qid)
                cached = await search_cache(question, document, chat_id)
                if cached:
                    logger.info(\"[%s]  [p0] cache HIT sim=%.2f\", _qid, cached.get(\"similarity\",0))
                else:
                    logger.info(\"[%s]  [p0] cache MISS\", _qid)
                if cached and cached[\"similarity\"] >= 0.92:
                    logger.info(\"[%s]  [p0] -> returning cached (%.0fms)\", _qid, (time.time()-_t0)*1000)
                    return {
                        \"answer\": cached[\"full_answer\"],
                        \"sources\": [{\"filename\": cached.get(\"source_doc\",\"Unknown\"),
                                     \"score\": cached[\"similarity\"]}],
                    }
            except Exception as e:
                logger.info(\"[%s]  [p0] cache error: %s\", _qid, e)

        # --- Phase 1"""

if old_p0 not in data:
    print("ERROR: P0 pattern not found")
    idx = data.find("Phase 0: Knowledge cache check")
    print(repr(data[idx:idx+250]))
    sys.exit(1)
data = data.replace(old_p0, new_p0, 1)

# ── P1: Add retrieval log ───────────────────────────────────
old_p1 = """        # --- Phase 1: Sync retrieval with document boosting + feedback ---
        retriever = self._get_retriever()
        nodes = retriever.retrieve(question)

        # Load feedback learner"""

new_p1 = """        # --- Phase 1: Sync retrieval with document boosting + feedback ---
        logger.info(\"[%s]  [p1] retrieve.start\", _qid)
        retriever = self._get_retriever()
        nodes = retriever.retrieve(question)
        _t1 = time.time()
        _n_chunks = len(nodes)
        _n_tokens = sum(len((n.text or \"\").split()) * 1.3 for n in nodes)
        _n_chars = sum(len(n.text or \"\") for n in nodes)
        _scores = [round(n.score or 0, 3) for n in nodes[:5]]
        _files = list({n.metadata.get(\"file_name\",\"?\") for n in nodes})
        logger.info(\"[%s]  [p1] retrieve.done %d nodes %d chars ~%d toks scores=%s files=%s (%.0fms)\",
                    _qid, _n_chunks, _n_chars, int(_n_tokens), _scores, _files, (_t1-_t0)*1000)

        # Load feedback learner"""

if old_p1 not in data:
    print("ERROR: P1 pattern not found")
    sys.exit(1)
data = data.replace(old_p1, new_p1, 1)

# ── P2: Add LLM phase log ───────────────────────────────────
old_p2 = """        # --- Phase 2: Async LLM generation (tree_summarize) ---
        chunk_texts = [n.text for n in nodes if n.text]
        
        # Log retrieval diagnostics
        total_tokens = sum(len(t.split()) * 1.3 for t in chunk_texts)
        logger.info(
            \"Retrieval: %d chunks, ~%d tokens (k=%s, ctx=%s)\",
            len(chunk_texts), int(total_tokens), settings.retrieval_top_k, 2048,
        )

        from app.rag.models import get_llm
        from app.rag.guard import OllamaGuard, OllamaBusyError, OllamaDeadError

        try:
            async with OllamaGuard(\"answer generation\", timeout=120):
                llm = get_llm()
                from llama_index.core.response_synthesizers import TreeSummarize
                synthesizer = TreeSummarize(llm=llm)
                response = await synthesizer.aget_response(
                    question,
                    chunk_texts,
                )
                answer = str(response)
        except OllamaBusyError as e:
            return {
                \"answer\": str(e),
                \"sources\": sources,
            }"""

new_p2 = """        # --- Phase 2: LLM generation (tree_summarize) ---
        chunk_texts = [n.text for n in nodes if n.text]
        _total_toks = sum(len(t.split()) * 1.3 for t in chunk_texts)
        logger.info(\"[%s]  [p2] LLM START chunks=%d ~%d tokens\", _qid, len(chunk_texts), int(_total_toks))

        from app.rag.models import get_llm
        from app.rag.guard import OllamaGuard, OllamaBusyError, OllamaDeadError

        try:
            async with OllamaGuard(\"answer generation\", timeout=120) as _guard:
                logger.info(\"[%s]  [p2] OllamaGuard acquired\", _qid)
                llm = get_llm()
                logger.info(\"[%s]  [p2] LLM model=%s\", _qid, getattr(llm, 'model', '?'))

                from llama_index.core.response_synthesizers import TreeSummarize
                synthesizer = TreeSummarize(llm=llm)
                logger.info(\"[%s]  [p2] TreeSummarize created, calling aget_response()...\", _qid)

                _t2 = time.time()
                response = await synthesizer.aget_response(question, chunk_texts)
                _gen_ms = (time.time()-_t2)*1000
                answer = str(response)
                logger.info(\"[%s]  [p2] LLM DONE %d chars (%.0fms gen, %.0fms total)\",
                            _qid, len(answer), _gen_ms, (time.time()-_t0)*1000)
        except OllamaBusyError as e:
            logger.error(\"[%s]  [p2] Ollama BUSY: %s\", _qid, e)
            return {\"answer\": str(e), \"sources\": sources}
        except OllamaDeadError as e:
            logger.error(\"[%s]  [p2] Ollama DEAD: %s\", _qid, e)
            return {\"answer\": \"⚠️ Study engine unavailable. Try again in 30s.\", \"sources\": sources}
        except Exception as e:
            logger.error(\"[%s]  [p2] LLM CRASHED: %s\", _qid, e, exc_info=True)
            raise"""

if old_p2 not in data:
    print("ERROR: P2 pattern not found")
    sys.exit(1)
data = data.replace(old_p2, new_p2, 1)

# ── P3: Update return to include _qid log ───────────────────
old_ret = """        return {
            \"answer\": answer,
            \"sources\": sources,
        }
    # ── Query with sources + document boosting ────────────"""

new_ret = """        _total_ms = (time.time()-_t0)*1000
        logger.info(\"[%s] >>> q_with_sources DONE %d chars (%.0fms total)\",
                    _qid, len(answer), _total_ms)

        return {
            \"answer\": answer,
            \"sources\": sources,
        }
    # ── Query with sources + document boosting ────────────"""

if old_ret not in data:
    print("ERROR: return pattern not found")
    sys.exit(1)
data = data.replace(old_ret, new_ret, 1)

with open(path, 'w') as f:
    f.write(data)
print("Done! Diagnostic logging injected into query_with_sources.")