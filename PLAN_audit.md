# Magic Grimoire — Audit: True Agentic Flow vs Current Reality

## Crash Diagnosis

**Symptom:** Bot died after `/ask "whats key takeaway from world economy book"`.

**Root cause:** Ollama crashed. No crash log — the llama runner process terminated silently (VRAM exhaustion from 2 concurrent operations). When Ollama crashed, ALL models disappeared from the Ollama store. `start-bot.sh` detected missing models and tried to re-pull 4.7 GB.

**The query DID work:** Looking at the log, Gemini correctly extracted the document name and the ask tool executed successfully with `200 OK`. But Ollama died shortly after (either during the response or the next operation).

**Fix:** Models need to persist across Ollama restarts. Need to investigate why models disappear on crash.

---

## The Vision vs Reality

### PLAN_agentic.md Vision

```
User free text
  ↓
Gemini Agent — decides tools, adapts, chains
  ├─ tool: ask/quiz/summarize/search → RAG → result
  ├─ tool: chat → Ollama
  └─ no tool → casual chat
  ↓
Agent sees result → decides next step → chains
```

**Key promises:**
- "chaining tools together"
- "adapting from feedback"  
- "feels like a real study partner"
- "too hard? → adapts"
- "tell me more → follow-up with context"

### Current Reality

```
User free text
  ↓
Gemini → ONE shot → TOOL: ask(...)
  ↓
Python executes tool (with hardcoded progress messages)
  ↓
Result sent to user
  ↓
END. No tool chaining. No feedback. No adaptation.
```

**Single-shot only.** Python never gives the result back to Gemini. There's no "agentic loop."

---

## 7 Gaps from True Agentic Flow

| # | Gap | How It Shows | Severity |
|---|---|---|---|
| 1 | **Single-shot only** | `"quiz me" → TOOL: quiz(topic)` → done. Gemini never sees the quiz result. Can't say "too hard? want easier?" | 🔴 Critical |
| 2 | **Python-generated progress** | `"📖 Searching documents..."` is hardcoded, not agent-driven. Violates core principle. | 🔴 Critical |
| 3 | **No multi-tool chaining** | `"search about X AND quiz me"` → Gemini includes both TOOL lines. Python executes both once. But agent never decides to chain based on intermediate results. | 🟡 Major |
| 4 | **No feedback loop for adaptation** | `"explain that more"` → Gemini recalls conversation history (last 5 exchanges) but never sees the RAG context or answer quality. | 🟡 Major |
| 5 | **Document matching fragile** | Gemini passes full EPUB filename `"The World Economy and Financial System A Paradigm Change Offering.epub"`. Metadata might store it differently. Need fuzzy match. | 🟠 Medium |
| 6 | **Ollama crash = model loss** | Models disappear when Ollama dies. Re-pull 4.7 GB each time. Need persistent model storage. | 🟠 Medium |
| 7 | **Conversation memory shallow** | `_remember()` stores only user text + final reply. No intermediate context (retrieved passages, scores, tool results). | 🟡 Minor |

---

## What IS Working

| Component | Status |
|---|---|
| Gemini intent routing | ✅ 50ms, reliable, good at tool selection |
| RAG pipeline (retrieve + generate) | ✅ Works, sync retrieval solid |
| Source citations | ✅ Shows sources with confidence |
| LiteParse + fallback parsing | ✅ Handles PDF, EPUB, scanned docs |
| File management | ✅ Upload, list, delete by ID |
| Progress timer | ✅ 60s keepalive during long ops |
| Error handling | ✅ Gateway timeout, tool try-except |
| Document score boosting | ✅ Implemented but fragile (gap #5) |
| Model warm-up | ✅ Pre-loads qwen2.5:7b on startup |

---

## The Approach: True Agentic Loop

### Architecture Change

**Before (current):**
```
Request → Gemini (1x) → Tool execution → Reply
```

**After (true agentic):**
```
Request → Gemini (turn 1) → message + TOOL: tool(params)
                ↓
            Python sends message to user (Gemini text)
                ↓
            Python executes tool → tool result
                ↓
            Python sends tool result back to Gemini (turn 2)
                ↓
          Gemini (turn 2) → "Here's what I found..." + optional follow-up TOOL
                ↓
            [loop continues until Gemini stops calling tools]
                ↓
            Final message sent to user
```

### Turn 1 Example

```
User: "what are key takeaways from world economy?"

Gemini (turn 1):
  "Let me search your World Economy document for key takeaways.\n\n"
  "TOOL: retrieve(query=\"key takeaways\", document=\"World Economy\")"
  
Python:
  1. Send "Let me search your World Economy document..." as progress message
  2. Execute retrieve(query="key takeaways", document="World Economy")
  3. Get back: [passages with scores]
```

### Turn 2 Example

```
Python feeds result back to Gemini:
  "Retrieved 3 passages from The World Economy [score: 0.92]:
   1. The global economy faces a paradigm shift from debt-driven...
   2. Key structural changes include...
   3. ..."

Gemini (turn 2):
  "Great, I found 3 relevant passages. Let me generate a concise answer.\n\n"
  "TOOL: generate_from_passages(question=\"...\", difficulty=\"normal\")"

Python:
  1. Edit message: "🧠 Generating answer..."
  2. Execute generate_from_passages → LLM generates answer
  3. Result: [full answer]
```

### Turn 3 Example

```
Python feeds result back:
  "Generated answer: The key takeaway from... [full text with sources]"

Gemini (turn 3):
  "Here are the key takeaways I found:
  
  [includes the generated answer]
  
  Want me to quiz you on this, or explain any part in detail?"

Python: Send final message to user. Loop ends.
```

Or Gemini might chain tools:
```
Gemini (turn 3):
  "TOOL: quiz(topic=\"world economy paradigm shift\", count=3)"
  
Python: Execute quiz → back to Gemini → ... → final message
```

---

## New Tool Architecture

### Tools (Reorganized)

| Tool | What It Does | Gemini Decides When |
|---|---|---|
| `retrieve(query, document?)` | Get raw passages (fast, no LLM) | Always first step |
| `generate_from_passages(question, passages, difficulty)` | LLM generates answer | After retrieval looks good |
| `quiz_from_passages(topic, passages, count)` | Generate quiz from passages | After retrieval on request |
| `summarize_from_passages(topic, passages)` | Generate summary from passages | After retrieval on request |
| `list_docs()` | List all indexed docs | "what docs do I have?" |
| `chat(text)` | Direct conversation (no RAG) | Greetings, casual talk |
| `check_answer_quality(answer, question)` | Gemini evaluates if answer is good | After generation (Gemini's own judgment) |

### Why Split retrieve + generate?

1. **Transparency**: User sees "🔍 Found 3 passages from World Economy" BEFORE waiting for LLM
2. **Agent control**: Gemini sees passages → can decide "these look good, generate" or "these are wrong, retry"
3. **Better error recovery**: Bad retrieval → Gemini can rephrase query and retry
4. **No hardcoded Python messages**: All messages are Gemini-generated

---

## Implementation Plan

### Phase 1: Agentic Loop Engine

Add `AgenticLoop` class in `app/plugins/brain/handler.py`:

```python
class AgenticLoop:
    MAX_TURNS = 5  # Safety limit
    
    async def run(self, user_text: str, chat_id: int, history: str) -> str:
        conversation = [{"role": "user", "text": user_text, "history": history}]
        last_message_id = None
        
        for turn in range(self.MAX_TURNS):
            # 1. Send to Gemini
            gemini_response = await self._call_gemini(conversation)
            
            # 2. Extract message + TOOL lines
            message, tools = self._parse_response(gemini_response)
            
            # 3. If no tools → conversation complete → send message
            if not tools:
                if last_message_id:
                    await self._edit(last_message_id, message)
                else:
                    await self._send(chat_id, message)
                return
            
            # 4. Send progress message (Gemini's text)
            if message:
                if last_message_id:
                    await self._edit(last_message_id, message)
                else:
                    msg_id = await self._send(chat_id, message + "\n⏳ Processing...")
                    last_message_id = msg_id
            
            # 5. Execute tools
            for tool_name, tool_params in tools:
                result = await self._execute_tool(tool_name, tool_params)
                # 6. Feed tool result as next turn
                conversation.append({
                    "role": "tool",
                    "tool": tool_name,
                    "result": result,
                })
        
        # Safety: max turns exceeded
        return "⚙️ Processing took too many steps. Please simplify your question."
```

### Phase 2: Agent Prompt Update

```
You are a study assistant with iterative tool access.

WORKFLOW:
1. For QUESTIONS: Use retrieve() first → see passages → then use generate() 
2. For QUIZZES: Use retrieve() → then quiz_from_passages()
3. For CHAT: Just respond, no tools needed
4. After generating: Always ask "want me to quiz you?" or "explain more?"

RULES:
- ALWAYS retrieve before generating — never assume what's in documents
- If retrieved passages look wrong (wrong doc, low scores): 
  retry with different query or tell the user
- NEVER produce the answer yourself for document questions — 
  always use retrieve + generate
- Your conversation text should feel natural and helpful
- You can use multiple turns for complex requests

TOOLS:
- retrieve(query, document?) → returns list of {filename, score, text_preview}
- generate_from_passages(question, passages_json, difficulty) → returns answer
- quiz_from_passages(topic, passages_json, count) → returns quiz
- summarize_from_passages(topic, passages_json) → returns summary
- list_docs() → returns list of {id, filename}
- chat(text) → returns conversational response from Ollama
```

### Phase 3: New Tool Implementations

```python
async def _tool_retrieve(query, document=None):
    """Get raw passages. Gemini sees them and decides next step."""
    return await retrieve_passages(query, document=document)

async def _tool_generate_from_passages(question, passages_json, difficulty="normal"):
    """Gemini passes retrieved passages directly. No re-retrieval needed."""
    passages = json.loads(passages_json)
    # Build context from passages, send to LLM
    context = "\n\n".join(p["text"] for p in passages[:5])
    # ... same LLM generation as current ask()

async def _tool_quiz_from_passages(topic, passages_json, count=5):
    """Generate quiz from pre-retrieved passages."""
    ...

async def _tool_summarize_from_passages(topic, passages_json):
    """Generate summary from pre-retrieved passages."""
    ...
```

### Files to Change

| File | Change |
|---|---|
| `app/plugins/brain/handler.py` | Add `AgenticLoop` class, new tools, updated prompt, replace current single-shot path |
| `app/rag/engine.py` | Keep as-is (retrieval + generation still used, but now called separately by tools) |
| `app/plugins/study/handler.py` | Keep `retrieve_passages()`, add `generate_from_passages()`, add `quiz_from_passages()` |

### Risk: Token Cost

Each turn costs:
- Gemini input: ~500 tokens (conversation + tool results)
- Gemini output: ~100 tokens
- Max 5 turns: ~3000 tokens total

Free tier: 1500 requests/day — well within limits for normal usage.

---

## Quick Fixes (While Building Agentic Loop)

These can be done NOW without the full loop:

| Fix | File | Effort |
|---|---|---|
| Gemini generates progress message | `brain/handler.py` — extract conversational text before TOOL line and send it as the thinking message | 15 min |
| Fuzzy document name matching | `engine.py` — use word-by-word matching instead of substring | 10 min |
| Ollama model persistence | Investigate why models disappear. Set `OLLAMA_MODELS` env var to a stable path | 10 min |
| Add `_tool_retrieve()` usage in slow-path | Already done ✅ in PLAN #5 | — |

---

## Priority

| Priority | What | Why |
|---|---|---|
| 🔴 P0 | Agentic Loop (Phase 1+2+3) | This IS the core vision. Without it, we have a smart router, not an agent. |
| 🟠 P1 | Gemini-generated progress messages | Violates core principle of agent-driven communication |
| 🟠 P1 | Fuzzy document matching | Current implementation works but fragile |
| 🟡 P2 | Ollama model persistence | Models shouldn't disappear on crash |
| 🟢 P3 | Deeper conversation memory | Store tool results alongside messages |
