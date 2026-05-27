# Magic Grimoire — PLAN: Progress Watcher Subprocess

## The Problem

Current progress ("⏳ Still generating... 60s elapsed") is meaningless. User doesn't know:
- Is it actually working or stuck?
- How much is left?
- What phase is it in?

## The Pattern: Watcher Task (asyncio, not OS subprocess)

A **parallel asyncio task** that reads shared state and sends intelligent updates. 
NOT an OS subprocess — that would require IPC, serialization, process management. 
Asyncio tasks share memory, are lightweight, and the bot already uses asyncio everywhere.

```
User query
  ↓
Main task: execute_operation()
  ├─ Phase 1: retrieve     → writes state: {"phase": "retrieve", "docs": 2}
  ├─ Phase 2: embed query  → writes state: {"phase": "embed"}
  ├─ Phase 3: generate     → writes state: {"phase": "generate", "tokens": 0}
  │                          → updates tokens as streaming progresses
  └─ DONE
       ↓
Watcher task: _watch_loop(interval=60)
  │  every 60s reads shared state → sends contextual update
  ├─ "Found 3 passages in World Economy. Writing answer... (60s)"
  ├─ "Still writing... 120 words so far (120s). Normal for detailed answers."
  └─ "Taking longer than expected (180s). Might retry with simpler query."
```

---

## What The Watcher Can Report

### Without streaming (today):

| Phase | State Available | Progress Report |
|---|---|---|
| `retrieve` | passage count, doc names, scores | "Found 3 passages from World Economy (0.92), Finance (0.45)" |
| `generate` | elapsed time only | "Writing answer... (60s). Normal for complex questions." |
| `generate` (late) | elapsed time > 120s | "Still working (120s). If this takes > 3 min, try a simpler question." |
| `index_parse` | file count, per-file status | "Parsed Finance.pdf (489p, 2.1s). Now on World Economy.epub..." |
| `index_embed` | batch progress, chunk count | "Embedding: 450/1200 chunks (38%). ~51s remaining." |

### With streaming (future):

| Phase | State Available | Progress Report |
|---|---|---|
| `generate` | token count, words written | "Writing: 85 words so far. Reading at ~3 words/s..." |
| `generate` (stall) | last token > 5s ago | "Model paused... resuming shortly." |

---

## Architecture: ProgressState + ProgressWatcher

### ProgressState (shared memory)

```python
@dataclass
class ProgressState:
    phase: str = "idle"          # retrieve | embed | generate | index_parse | index_embed
    started_at: float = 0.0
    phase_started_at: float = 0.0
    
    # Phase-specific data
    passages_found: int = 0
    passages_docs: list[str] = field(default_factory=list)
    tokens_generated: int = 0
    words_generated: int = 0
    chunks_embedded: int = 0
    chunks_total: int = 0
    files_parsed: int = 0
    files_total: int = 0
    
    # Status
    error: str | None = None
    complete: bool = False
```

### ProgressWatcher (asyncio task)

```python
class ProgressWatcher:
    def __init__(self, chat_id: int, message_id: int, 
                 state: ProgressState, interval: int = 60):
        self.chat_id = chat_id
        self.message_id = message_id
        self.state = state
        self.interval = interval
        self._task: asyncio.Task | None = None
        self._reports_sent: int = 0
    
    def start(self):
        self._task = asyncio.create_task(self._watch_loop())
    
    def stop(self):
        if self._task:
            self._task.cancel()
    
    async def _watch_loop(self):
        """Send phase-aware updates every `interval` seconds."""
        try:
            while True:
                await asyncio.sleep(self.interval)
                if self.state.complete:
                    return
                
                msg = self._build_report()
                if msg:
                    await _edit_message(self.chat_id, self.message_id, msg)
                    self._reports_sent += 1
        except asyncio.CancelledError:
            pass
    
    def _build_report(self) -> str | None:
        """Build contextual report based on current phase."""
        elapsed = int(time.time() - self.state.phase_started_at)
        total = int(time.time() - self.state.started_at)
        
        if self.state.phase == "generate":
            if self.state.tokens_generated > 0:
                # Streaming available
                return (
                    f"Writing answer: {self.state.words_generated} words "
                    f"({elapsed}s elapsed)"
                )
            else:
                # No streaming — just elapsed time with escalating tone
                if elapsed < 60:
                    return None  # Don't nag before 60s
                elif elapsed < 120:
                    return (
                        f"Still writing your answer... ({elapsed}s). "
                        "Normal for detailed questions."
                    )
                elif elapsed < 180:
                    return (
                        f"Taking longer than usual ({elapsed}s). "
                        "Complex topic? The model is processing carefully."
                    )
                else:
                    return (
                        f"Very long generation ({elapsed}s). "
                        "If this persists, try: /index (refresh index) "
                        "or ask a simpler question."
                    )
        
        elif self.state.phase == "index_embed":
            done = self.state.chunks_embedded
            total_c = self.state.chunks_total
            if total_c and done:
                pct = done * 100 // total_c
                rate = done / elapsed if elapsed > 0 else 0
                remaining = int((total_c - done) / rate) if rate > 0 else "?"
                return (
                    f"Embedding: {done}/{total_c} chunks ({pct}%). "
                    f"~{remaining}s remaining."
                )
            else:
                return f"Embedding chunks ({elapsed}s elapsed)..."
        
        elif self.state.phase == "index_parse":
            return (
                f"Parsing: {self.state.files_parsed}/{self.state.files_total} "
                f"files ({elapsed}s elapsed)"
            )
        
        return f"Still working... ({elapsed}s, {total}s total)"
```

---

## How It Wires In

### BrainPlugin slow-path:

```python
# Create shared state
progress = ProgressState()
progress.start_phase("retrieve")

# Start watcher
watcher = ProgressWatcher(chat_id, thinking_id, progress, interval=60)
watcher.start()

try:
    # Phase 1: Retrieve
    passages = await retrieve(query, document)
    progress.passages_found = len(passages)
    progress.passages_docs = [p["filename"] for p in passages[:3]]
    
    # Phase 2: Generate
    progress.start_phase("generate")
    answer = await generate(query, passages)
    
    # Done
    progress.complete = True
finally:
    watcher.stop()
```

### StudyPlugin /index:

```python
progress = ProgressState()
progress.start_phase("index_parse")
progress.files_total = len(files)

watcher = ProgressWatcher(chat_id, msg_id, progress, interval=30)
watcher.start()

try:
    for i, fname in enumerate(files):
        progress.files_parsed = i + 1
        parse_file(fname)
    
    progress.start_phase("index_embed")
    progress.chunks_total = len(documents)
    
    for batch in batches:
        embed_batch(batch)
        progress.chunks_embedded += len(batch)
    
    progress.complete = True
finally:
    watcher.stop()
```

---

## Escalation Policy

The watcher changes its tone based on elapsed time:

| Time | Frequency | Tone |
|---|---|---|
| 0-60s | Silent | Let it work |
| 60-120s | Every 60s | Reassuring: "Normal for detailed questions" |
| 120-180s | Every 60s | Cautious: "Taking longer than usual" |
| 180s+ | Every 30s | Concerned: "Try a simpler question or /index" |
| 300s+ | Stop | "Operation timed out" (gateway timeout kicks in) |

---

## What Makes This Different From Current Timer

| | Current `_progress_timer` | New `ProgressWatcher` |
|---|---|---|
| Reports | "⏳ Still generating... (60s)" | "Writing: 85 words. Normal for complex questions." |
| Phase-aware | No — just counts | Yes — different messages per phase |
| Escalation | No | Yes — tone changes at 60s, 120s, 180s |
| Data-driven | No data | token count, chunk progress, file count |
| Subprocess | No (asyncio task) | Same — asyncio task, not OS process |
| Overhead | Near zero | Near zero (reads a dataclass, no IPC) |

---

## Implementation Plan

### Phase 1: ProgressState + ProgressWatcher
- Create `app/ui/progress.py` with `ProgressState` and `ProgressWatcher`
- Wire into BrainPlugin slow-path for `/ask`
- Wire into StudyPlugin for `/index`

### Phase 2: Streaming integration (future)
- When streaming is implemented, update `tokens_generated` / `words_generated` during generation
- Watcher automatically picks up the data — no code change needed

### Files

| File | Action |
|---|---|
| `app/ui/__init__.py` | **NEW** — package init |
| `app/ui/progress.py` | **NEW** — ProgressState, ProgressWatcher |
| `app/plugins/brain/handler.py` | Wire watcher into slow-path |
| `app/plugins/study/handler.py` | Wire watcher into `_handle_index` |

### Why asyncio Task, Not OS Subprocess

```
OS subprocess:
  User query → fork() → child process → IPC (pipes/queues) → read state
  ❌ Overhead: process spawn (~50ms), serialization, fd management
  ❌ Complexity: two Python interpreters, two memory spaces
  ❌ Risk: orphan processes, zombie processes, signal handling

Asyncio task:
  User query → create_task(watcher) → shared memory (dataclass) → read state
  ✅ Overhead: near zero (just a coroutine on the event loop)
  ✅ Simplicity: one Python process, one memory space, no IPC
  ✅ Safety: task.cancel() cleans up instantly, no orphans
```

Same effect — "another thing watching and reporting" — without the subprocess overhead.
