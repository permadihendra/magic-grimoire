# Magic Grimoire — Development Plan

> Canonical development roadmap. Completed items stay marked ✅ for reference.
> In-progress items marked 🔨. Planned items show priority.

## Status: All phases complete ✅

This project is in **maintenance mode**. Focus areas:
- Stability on 8GB GPU (Ollama crash prevention, CPU fallback)
- Accurate document parsing (multi-method, pre-index verify)
- Consistent UI across all commands (source citations, metadata, diagnostics)
- Performance optimization (GPU utilization, timeout tuning)

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
- `engine.py` — `query_with_sources()` with 7 guardrails + TreeSummarize
- `prompts.py` — QA / Quiz / Summary / QnA with difficulty variants
- `guard.py` — OllamaGuard: VRAM pre-check + semaphore + OOM recovery
- `feedback.py` — FeedbackLearner: per-chat document penalty/boost

### ✅ Phase 3: Agentic Brain (Gemini)
- BrainPlugin intercepts all free text
- `AGENT_PROMPT` — tool definitions, rules, format examples (ask, quiz, summarize, **qna**, list_docs, chat, feedback, retrieve)
- `_TOOL_REGISTRY` — 8 tools registered
- `_SLOW_TOOLS` — ask/quiz/summarize/qna trigger thinking indicator
- `_conversation_memory` — last 5 exchanges per chat
- Detection footer — shows "ℹ️ from training data" when no TOOL used

### ✅ Phase 4: UX & Progress
- Warm-up on startup (2-token request → pre-loads LLM into VRAM)
- ProgressWatcher — phase-aware, sends Telegram edits at 0.5s, 30s, 60s, 120s+
- Knowledge cache — Q&A pairs with cosine similarity, bypasses RAG on hit ≥ 0.92
- Multi-message splitting — `_split_into_chunks()` for replies > 4000 chars
- Follow-up footer sent as separate message (next-step + diagnostics)
- Display names — Gemini cleans filenames at index time

### ✅ Phase 5: Stability (Critical)
- Fresh sqlite3 per operation — no background-thread corruption during GPU load
- FallbackEmbedding — Ollama embed crash → auto-switch to CPU all-MiniLM-L6-v2
- `embed_batch_size=3` — prevents Ollama Go runner crash
- OllamaGuard `Semaphore(1)` — only 1 Ollama operation at a time
- `_tool_chat()` wrapped with OllamaGuard (was unguarded)
- `ollama_check_vram()` uses async subprocess (was blocking event loop)
- PDF parsing timeout (30s per method) + LiteParse for OCR

### ✅ Phase 6: Index Verification
- `pre_index_check_single()` — test-parse each file before building
- `parse_document_multimethod()` — tries 4 parsers in order (ebooklib → LiteParse → Calibre → SimpleDirectoryReader)
- `probe_index()` — test retrieval after build to confirm usability
- `parse_method` + `verified` columns in DB
- Per-file report with parse method, word count, chunks, probe status

### ✅ Phase 7: Command UI Audit
- `_build_file_card()` — consistent document card (size, words, chunks, parse method, verified)
- `_build_files_card()` — file list with indexed status + display name from DB
- `_build_total_footer()` — summary with word/chunk totals
- `/docs` → word counts, parse methods, probe verified flags
- `/files` → display names, indexed vs pending, chunk counts
- `/delete` → impact report (size, words, chunks) before confirming
- `/quiz` + `/summarize` → source citations + next-step + retrieval stats

### ✅ Phase 8: Performance Optimization
- **100% GPU utilization** — `num_gpu 99` (all layers on GPU, was 1)
- **Power efficiency** — MSI Afterburner 80% power cap (host-side)
- **Synthesis timeout** — increased to 120s (was 30s)
- **OllamaGuard timeout** — increased to 180s (was 120s)
- **num_predict** — 2048 (was 128 from Modelfile, now sent via API)
- **num_ctx** — 2048 via Modelfile, 4096 via API `additional_kwargs`
- **Generation speed** — ~40 tok/s (was ~10-15 tok/s with num_gpu 1)

---

## Current Resource Profile

| Component | Model | VRAM | Notes |
|---|---|---|---|
| LLM | `magic-grimoire:3b` (qwen2.5:3b Q4_0) | 1.9 GB | All layers on GPU (`num_gpu 99`) |
| KV cache | 2048 context window | ~2.4 GB | Managed by Ollama |
| Embeddings | `nomic-embed-text` (batch=3) | 0.3 GB | Falls back to CPU `all-MiniLM-L6-v2` |
| **Total** | | **~4.6 GB** | 3.4 GB headroom on 8 GB |
| Power | GPU capped at 80% via MSI Afterburner | ~100W | Safe for 24/7 operation |

### Key Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `num_gpu` | 99 | All layers on GPU for max throughput |
| `num_ctx` (Modelfile) | 2048 | KV cache budget |
| `num_ctx` (API) | 4096 | Extra headroom via `additional_kwargs` |
| `num_predict` | 2048 | Max output tokens (~50s at 40 tok/s) |
| `temperature` | 0.0 | Deterministic output |
| Synthesis timeout | 120s | Max LLM generation wait |
| OllamaGuard timeout | 180s | Outer safety boundary |
| Gateway timeout | 300s | Dispatch-level boundary |

---

## Commands

| Command | Handler | Description |
|---|---|---|
| `/ask <q>` | `StudyPlugin._handle_ask()` | RAG query with source citations |
| `/quiz <topic>` | `StudyPlugin._handle_quiz()` | Practice questions with answers |
| `/summarize <topic>` | `StudyPlugin._handle_summarize()` | Topic summary |
| `/qna <topic>` | `StudyPlugin._handle_qna()` | Comprehension Q&A pairs (default 10) |
| `/docs` | `StudyPlugin._handle_docs()` | Document library with metadata |
| `/index` | `StudyPlugin._handle_index()` | Pre-check → parse → embed → probe |
| `/files` | `StudyPlugin._handle_files()` | All files on disk with indexed status |
| `/delete <id>` | `StudyPlugin._handle_delete()` | Delete file + show impact |
| Free text | `BrainPlugin.handle()` | Gemini routes to appropriate tool |

---

## Future Improvements

### 🟡 Streaming output (high priority)
Token-by-token LLM output to Telegram via `editMessageText`.
Currently: user waits for full generation → single edit.

### 🟡 Multi-turn agentic loop (high priority)
Gemini sees tool output and can chain → retry with feedback, follow-up questions,
combined ask+quiz responses.

### 🟢 Hybrid search (medium priority)
BM25 + vector for keyword-heavy queries (e.g. exact chapter/section names).

### 🟢 LLM reranker (medium priority)
Small model (cross-encoder) reranks retrieved chunks before generation.

### 🔴 Upload via Telegram (low priority)
Send PDF/EPUB directly in chat → bot saves to docs/ and triggers `/index`.
Currently: manual file copy to `app/docs/`.

---

## Closed Issues (for reference)

| Issue | Root Cause | Fix |
|---|---|---|
| WSL crash during `/index` | Ollama Go runner crashed on batch embed → GPU passthrough panic | embed_batch_size=3 + FallbackEmbedding |
| "unknown error" in `/index` result | `file_name` metadata stored as full path | `os.path.basename()` everywhere |
| DB corruption on index rebuild | aiosqlite background thread writing during GPU load | Fresh sqlite3 per operation + WAL |
| FTS5 DELETE corrupting DB | FTS5 DELETE conflicts with main table WAL | Removed FTS5, used keyword matching |
| Empty index → TreeSummarize hang | No chunks to synthesize → LlamaIndex bug | L1: 0 chunks → early return |
| `num_predict` silently ignored | LlamaIndex Ollama class swallows `num_predict` kwarg | Use `additional_kwargs` dict |
| Slow generation (~15 tok/s) | `num_gpu 1` — only 1 of ~24 layers on GPU | `num_gpu 99` — all layers on GPU (~40 tok/s) |
| `/qna` timeout | `llm.acomplete()` with 3000+ token prompt | Use `TreeSummarize` via `query_with_sources(mode="qna")` |
| Follow-up sent before answer | `_tool_ask()` sent follow_up via `sendMessage` before returning | Return `(answer, follow_up)` tuple, brain handler sends last |
| PDF parsing hangs | `SimpleDirectoryReader` could hang on scanned PDFs | 30s timeout per parser method + LiteParse for OCR |
| Follow-up footer never sent | `ask_query()` discarded dict fields, returned only string | Return dict with `follow_up`, handled by gateway/brain handler |
| `_tool_chat()` concurrent crash | Direct Ollama call without OllamaGuard | Wrapped with `OllamaGuard` |
| `nvidia-smi` blocking event loop | `subprocess.run()` in async context | Switched to `asyncio.create_subprocess_exec()` |
| `/qna` only 1 pair shown | Strict `[N] Q:` parsing failed on TreeSummarize output | Raw output display + regex counting |
| `partial_format` error | Plain string passed as `summary_template` to TreeSummarize | Wrap in `PromptTemplate` object |
