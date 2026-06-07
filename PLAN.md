# Magic Grimoire — Development Plan

> Canonical development roadmap. Completed items stay marked ✅ for reference.
> In-progress items marked 🔨. Planned items show priority.

## Status: All phases complete ✅ — OpenClaw integration live ✅

---

## ✅ Phase 7: OpenClaw Integration (NEW)

- REST API at `/api/tools/*` — exposes all study tools as HTTP endpoints
- OpenClaw skill at `skills/magic-grimoire/SKILL.md` — auto-discovered by OpenClaw
- Document filtering — `document` param on quiz/summarize/qna/ask endpoints
- Adaptive Q&A mode — trivia (default) vs comprehension (keyword-triggered)
- Context-aware follow-ups — API returns `context` field for next query
- Agent-level query refinement — skill teaches OpenClaw to expand vague queries
- Symlink install — `~/.openclaw/skills/magic-grimoire → repo/skills/magic-grimoire`
- Service fallback — skill auto-starts service if unreachable

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
- `engine.py` — `query_with_sources()` with 7 guardrails + TreeSummarize + 120s timeout
- `prompts.py` — QA / Quiz / Summary / QnA with difficulty variants
- `guard.py` — OllamaGuard: VRAM pre-check (async) + semaphore + OOM recovery
- `feedback.py` — FeedbackLearner: per-chat document penalty/boost

### ✅ Phase 3: Agentic Brain (Gemini)
- BrainPlugin intercepts all free text
- `AGENT_PROMPT` — tool definitions, rules, format examples
- `_TOOL_REGISTRY` — 9 tools: ask, quiz, summarize, qna, list_docs, chat, feedback, retrieve, delete
- `_SLOW_TOOLS` — ask/quiz/summarize/qna trigger thinking indicator
- `_conversation_memory` — last 5 exchanges per chat
- Detection footer — shows "ℹ️ from training data" when no TOOL used

### ✅ Phase 4: UX & Progress
- Warm-up on startup (2-token request → pre-loads LLM into VRAM)
- ProgressWatcher — phase-aware, sends Telegram edits at 0.5s, 30s, 60s, 120s+
- Knowledge cache removed — fresh generation every time
- `_safe_truncate()` — 4000 char Telegram safety
- Display names — Gemini cleans filenames at index time

### ✅ Phase 5: Stability (Critical)
- Fresh sqlite3 per operation — no background-thread corruption during GPU load
- FallbackEmbedding — Ollama embed crash → auto-switch to CPU all-MiniLM-L6-v2
- embed_batch_size=3 — was 10, too many concurrent calls crashed Go runner
- VRAM pre-check before every Ollama call (OllamaGuard) — async nvidia-smi
- TreeSummarize 120s hard timeout → fallback simple prompt
- `num_ctx=4096` — enough headroom for long answers
- `temperature=0.1` — slight variation for repeated questions
- `_tool_chat()` wrapped in OllamaGuard — prevents concurrent Ollama calls

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

### ✅ Phase 8: QnA Command
- `/qna` slash command + Gemini tool routing
- Generic prompt examples (no content anchoring)
- `generate_qna()` module-level function for brain handler
- Pre-retrieval preview (passage names shown before generation)

### ✅ Phase 9: Freshness & Cache Removal
- Knowledge cache removed from `engine.py` — no stale answers
- Temperature 0.0 → 0.1 — slight variation across sessions
- Prompt templates updated with "vary between sessions" instructions
- `_qna_history` removed — every `/qna` call is independent

### ✅ Phase 10: Safety Fixes
- `_tool_chat()` wrapped in OllamaGuard — prevents concurrent Ollama calls
- `ollama_check_vram()` uses `asyncio.create_subprocess_exec` — no event loop blocking
- Follow-up ordering fixed — `_tool_ask` returns tuple, brain handler sends follow-up last
- Timeout chain updated: OllamaGuard 180s → TreeSummarize 120s → fallback 60s

---

## 🔨 In Progress / Known Issues

### `num_gpu=99` in Modelfile — Should Be 1
`Modelfile.3b` has `PARAMETER num_gpu 99`. For Q4_0 quantized models, this should be `1`.
`99` is for layered models (e.g., GGUF with many layers). Quantized models use single-layer GPU.
**Impact:** May cause Ollama to try loading too many GPU layers → potential OOM on 8GB.
**File:** `Modelfile.3b:12`

### `num_predict=2048` — May Be Too Large
Currently 2048 tokens max output. At ~40 tok/s, that's ~50s generation.
`num_ctx=4096` + `num_predict=2048` = large KV cache + long output = more VRAM pressure.
**Consideration:** If VRAM is tight, reduce to 1024. If stable, keep at 2048.

### TOCTOU Race in OllamaGuard
`_ollama_sem.locked()` check before `acquire()` has a race window.
Low risk — `acquire()` is atomic, worst case is a slightly delayed "busy" message.
**File:** `app/rag/guard.py:141-144`

---

## Future Improvements

### ✅ RAG v2 — Retrieval Quality Overhaul (highest impact)
Three new pipeline stages added to `query_with_sources()`:

1. **Query Refiner (Gemini)** — strip filler, expand keywords (~50ms)
2. **Over-Retrieve + Cross-Encoder Reranker** — top 20 → rerank → top 5 (~0.5GB VRAM)
3. **BM25 Fallback** — keyword search when rerank returns empty

**Files:** `app/rag/reranker.py`, `app/rag/query_refiner.py`, `app/rag/bm25_fallback.py`
**Config:** `use_query_refiner`, `use_reranker`, `use_bm25_fallback` — all in `config.py`
**Rollback:** Set all three to `False` to revert to old pipeline

### 🟡 Streaming Output (high priority)
Token-by-token LLM output to Telegram via `editMessageText`.
Currently: user waits for full generation → single edit.
Target: tokens appear progressively.

### 🟡 Multi-turn Agentic Loop (high priority)
Gemini sees tool output and can chain → retry with feedback, follow-up questions,
combined ask+quiz responses.

### 🟢 Hybrid Search (medium priority)
BM25 + vector for keyword-heavy queries (e.g. exact chapter/section names).
Pairs with RAG v2's BM25 fallback.

### 🟢 Better Filename Cleaning (medium priority)
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
| Gemini ignoring RAG tools | "Helpful but lazy" — answers from training data | Detection footer + stronger prompt |
| `/qna` shows 1 pair instead of 10 | Strict parser fails on model output variations | Raw output display + pattern counting |
| QnA always returns Krishna content | Hardcoded examples in QNA_PROMPT | Generic placeholder examples |
| Same answer every time | Knowledge cache + temperature 0.0 | Cache removed, temp 0.1, prompt variation |
| `_tool_chat` bypasses guard | Direct Ollama call without OllamaGuard | Wrapped in OllamaGuard |
| `subprocess.run` blocks event loop | Synchronous nvidia-smi call | Switched to asyncio.create_subprocess_exec |
| Follow-up sent before answer | `_tool_ask` sent follow-up via Telegram before returning | Tuple return, brain handler sends last |

---

## VRAM Budget (RTX 3050 8GB)

| Component | VRAM |
|---|---|
| LLM weights (magic-grimoire:3b) | 1.9 GB |
| KV cache (4096 ctx) | ~2.4 GB |
| Ollama embed (nomic-embed-text) | 0.3 GB |
| **Total used** | **~4.6 GB** |
| **Headroom** | **~3.4 GB** |

CPU fallback (sentence-transformers) uses 0 VRAM — available when Ollama embed fails.
Cross-encoder reranker (if added): ~0.5 GB — fits within headroom.

---

## Architecture: Message Flow

```
Telegram message
  ↓
gateway.py (POST /telegram) → parse update
  ↓
dispatcher.py (dispatch update)
  ├─ Slash command → Plugin.handle() by command
  └─ Free text    → BrainPlugin.handle() [Gemini agent]
        ↓
        Gemini classifies intent + decides tools
        ↓
        ┌─ ask()      → study/handler.ask_query() → engine.query_with_sources()
        ├─ quiz()     → study/handler.generate_quiz()
        ├─ summarize()→ study/handler.summarize_topic()
        ├─ qna()      → study/handler.generate_qna()
        ├─ list_docs()→ brain/handler._tool_list_docs_v2()
        ├─ feedback() → rag/feedback.py FeedbackLearner
        ├─ retrieve() → rag/engine.retrieve_only()
        └─ chat()     → Ollama direct (guarded by OllamaGuard)
              ↓
        Response edited into Telegram "thinking" message
```

---

## Common Commands

```bash
bash app/start-bot.sh            # Always use this — manages cloudflared tunnel
bash app/stop-bot.sh             # Kill all bot processes cleanly
bash app/dependency-check.sh     # Verify all deps (--fix to auto-install)
uv sync                          # Install deps
uv sync --extra epub             # Install EPUB support (ebooklib + html2text)
uv sync --extra dev              # Install dev deps
uv run ruff check .              # Lint

ollama list                      # Check models
ollama show magic-grimoire:3b    # Inspect model info
nvidia-smi                       # Check VRAM

# Register webhook (one-time)
uv run python -m app.bot.setup_webhook
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Bot doesn't reply | Re-run `bash app/start-bot.sh` (cloudflared tunnel expired) |
| "Model not found" | `ollama create magic-grimoire:3b -f Modelfile.3b` |
| Index rebuild fails | `ollama serve` + `ollama list` to verify models |
| "Ollama embed Go runner crashed" | Bot auto-switches to CPU fallback — retry `/index` |
| "database disk image is malformed" | Delete `data/magic-grimoire.db` and restart |
| PC crashed during `/index` | Install `sentence-transformers` for CPU fallback |
| Wrong source cited | Say "that's from the wrong book" → feedback penalizes it |
