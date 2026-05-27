# Magic Grimoire — Development Plan

> Canonical development roadmap. Completed items stay marked ✅ for reference.
> In-progress items marked 🔨. Planned items show priority.

## Status: All phases complete ✅

This project is in **maintenance mode**. Focus areas:
- Stability on 8GB GPU (Ollama crash prevention, CPU fallback)
- Accurate document parsing (multi-method, pre-index verify)
- Consistent UI across all commands (source citations, metadata, diagnostics)

---

## Implementation Log

### ✅ Phase 1: Foundation
- Project scaffolded (FastAPI, LlamaIndex, Ollama, Gemini)
- `database.py` — sqlite3 fresh-per-op (WAL mode), 3 tables
- `config.py` — pydantic-settings with all secrets
- `start-bot.sh` — cloudflared tunnel + Ollama health check + warm-up
- Dispatcher routes: slash commands → Plugin, free text → BrainPlugin

### ✅ Phase 2: RAG Core
- `models.py` — Ollama LLM (magic-grimoire:3b) + FallbackEmbedding
- `indexer.py` — multi-method parsing (ebooklib → LiteParse → Calibre → SimpleDirectoryReader)
- `engine.py` — `query_with_sources()` with 7 guardrails + TreeSummarize + 30s timeout
- `prompts.py` — QA / Quiz / Summary with difficulty variants
- `guard.py` — OllamaGuard: VRAM pre-check + semaphore + OOM recovery
- `feedback.py` — FeedbackLearner: per-chat document penalty/boost

### ✅ Phase 3: Agentic Brain (Gemini)
- BrainPlugin intercepts all free text
- `AGENT_PROMPT` — tool definitions, rules, format examples
- `_TOOL_REGISTRY` — 8 tools: ask, quiz, summarize, list_docs, chat, feedback, retrieve, delete
- `_SLOW_TOOLS` — ask/quiz/summarize trigger thinking indicator
- `_conversation_memory` — last 5 exchanges per chat
- Detection footer — shows "ℹ️ from training data" when no TOOL used

### ✅ Phase 4: UX & Progress
- Warm-up on startup (2-token request → pre-loads LLM into VRAM)
- ProgressWatcher — phase-aware, sends Telegram edits at 0.5s, 30s, 60s, 120s+
- Knowledge cache — Q&A pairs with cosine similarity, bypasses RAG on hit ≥ 0.92
- `_safe_truncate()` — 4000 char Telegram safety
- Display names — Gemini cleans filenames at index time

### ✅ Phase 5: Stability (Critical)
- Fresh sqlite3 per operation — no background-thread corruption during GPU load
- FallbackEmbedding — Ollama embed crash → auto-switch to CPU all-MiniLM-L6-v2
- embed_batch_size=3 — was 10, too many concurrent calls crashed Go runner
- VRAM pre-check before every Ollama call (OllamaGuard)
- TreeSummarize 30s hard timeout → fallback simple prompt
- `num_ctx=1024` (was 2048) — halves KV cache, more headroom
- `num_gpu=1` (was 99) — explicit for Q4_0 quantized model
- `num_predict=128` (was 256) — limits response length

### ✅ Phase 6: Index Verification
- `pre_index_check_single()` — test-parse each file before building
- `parse_document_multimethod()` — tries 4 parsers in order
- `probe_index()` — test retrieval after build to confirm usability
- `parse_method` + `verified` columns in DB
- Block on all-fail — never says "success" when 0 docs parsed
- Per-file report with parse method, word count, chunks, probe status
- `os.path.basename` for consistent file_name metadata

### ✅ Phase 7: Command UI Audit
- `_build_file_card()` — consistent document card (size, words, chunks, parse method, verified)
- `_build_files_card()` — file list with indexed status + display name from DB
- `_build_total_footer()` — summary with word/chunk totals
- `/docs` → word counts, parse methods, probe verified flags
- `/files` → display names, indexed vs pending, chunk counts
- `/delete` → impact report (size, words, chunks) before confirming
- `/quiz` + `/summarize` → source citations + next-step + retrieval stats (same pattern as `/ask`)

---

## Future Improvements

### 🟡 Streaming output (high priority)
Token-by-token LLM output to Telegram via `editMessageText`.
Currently: user waits for full generation → single edit.
Target: tokens appear progressively.

### 🟡 Multi-turn agentic loop (high priority)
Gemini sees tool output and can chain → retry with feedback, follow-up questions,
combined ask+quiz responses.

### 🟢 Hybrid search (medium priority)
BM25 + vector for keyword-heavy queries (e.g. exact chapter/section names).

### 🟢 LLM reranker (medium priority)
Small model (cross-encoder) reranks retrieved chunks before generation.

### 🟢 Better filename cleaning (medium priority)
Gemini filename cleaner → fallback to `_simple_clean_filename()` if Gemini fails.
Currently: raises on Gemini error, falls back to simple.

### 🔴 Upload via Telegram (low priority)
Send PDF/EPUB directly in chat → bot saves to docs/ and triggers `/index`.
Currently: manual file copy to `app/docs/`.

---

## Closed Issues (for reference)

| Issue | Root Cause | Fix |
|---|---|---|
| WSL crash during `/index` | Ollama Go runner crashed on batch embed → GPU passthrough panic | embed_batch_size=3 + FallbackEmbedding |
| "unknown error" in `/index` result | `file_name` metadata stored as full path, compared against basename | `os.path.basename()` everywhere |
| "table has no column word_count" | New column not in existing DB | ALTER TABLE ADD COLUMN in init_db |
| "Can't instantiate abstract class FallbackEmbedding" | Missing `_aget_query_embedding`, `_get_query_embedding` | Added both methods |
| DB corruption on index rebuild | aiosqlite background thread writing during GPU load | Fresh sqlite3 per operation + WAL |
| FTS5 DELETE corrupting DB | FTS5 DELETE conflicts with main table WAL | Removed FTS5, used keyword matching |
| Empty index → TreeSummarize hang | No chunks to synthesize → LlamaIndex bug | L1: 0 chunks → early return |

---

## VRAM Budget (RTX 3050 8GB)

| Component | VRAM |
|---|---|
| LLM weights (magic-grimoire:3b) | 1.9 GB |
| KV cache (1024 ctx) | ~1.2 GB |
| Ollama embed (nomic-embed-text) | 0.3 GB |
| **Total used** | **~3.4 GB** |
| **Headroom** | **~4.6 GB** |

CPU fallback (sentence-transformers) uses 0 VRAM — available when Ollama embed fails.