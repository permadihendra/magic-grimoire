# Magic Grimoire — Phase 5: Agentic Brain (Option C)

## Vision

Give Magic Grimoire an **agentic brain** powered by Gemini that understands user intent, chains tools together, adapts from feedback, and feels like a real study partner — not a menu-driven bot.

```
User free text
  ↓
Gemini Agent (BrainPlugin)
  ├─ tool: ask(query)        → StudyPlugin → RAG → answer
  ├─ tool: quiz(topic, n)    → StudyPlugin → RAG → questions
  ├─ tool: summarize(topic)  → StudyPlugin → RAG → summary
  ├─ tool: search(query)     → StudyPlugin → RAG → passages (no LLM)
  ├─ tool: chat(text)        → direct Ollama conversation
  └─ no tool                 → casual chat reply
  ↓
Natural response with tool results embedded
```

---

## Why Gemini for the Brain?

| Criteria | Gemini (cloud) | Ollama (local) |
|---|---|---|
| Intent classification | **50ms** ⚡ | 5-10s 🐢 |
| Cost | **Free tier** (1500 req/day) | Free but slow |
| Tool calling | Native via prompt | Needs function calling setup |
| Follow-up context | Natural | Possible but slower |
| Reliability | Google infra | Depends on GPU load |

Gemini handles the **lightweight thinking** (intent, routing, clarifying). Ollama handles the **heavy lifting** (RAG generation, quizzes).

---

## Architecture

### Before (current)
```
Dispatcher → Free text → StudyPlugin.handle()
                           ↓
                         RAG Engine → Ollama
```

### After (Phase 5)
```
Dispatcher → Free text → BrainPlugin (Gemini Agent)
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       "ask"          "quiz"          "chat"
            │              │              │
         StudyPlugin   StudyPlugin    Ollama (direct)
         .ask()        .quiz()        conversation
            │              │
         RAG Engine    RAG Engine
            │              │
         Ollama         Ollama
```

### Gemini Agent Prompt

```
You are a study assistant agent. You have access to tools.
Respond conversationally. When you need to use a tool,
include TOOL: lines in your response.

Tools:
- TOOL: ask(query)          → query documents, get answer
- TOOL: quiz(topic, n=5)    → generate n practice questions
- TOOL: summarize(topic)    → create topic summary
- TOOL: search(query)       → find relevant passages (fast)

Rules:
- For greetings/chat → just respond, no tool needed
- For "what is X" → use ask()
- For "quiz me" → use quiz()
- For "explain X" → use ask()
- For "tell me more" → use ask() with context from conversation
- For "too hard/easy" → use quiz(adjusted) to adapt
- For ambiguous requests → use ask() + natural clarification
- You can chain multiple TOOL: lines if needed
- Include tool at the END of your response
```

---

## Tool Definitions

### 1. `ask(query)` — Document Q&A
```
Input:  Natural language question
Action: StudyPlugin._ask(query) → RAG engine → answer
Output: Answer text (replaces TOOL: line)
```

### 2. `quiz(topic, n=5)` — Generate Questions
```
Input:  Topic + optional count (default 5)
Action: StudyPlugin._quiz(topic, n) → RAG engine (quiz mode)
Output: Quiz questions with answer key
```

### 3. `summarize(topic)` — Topic Summary
```
Input:  Topic to summarize
Action: StudyPlugin._summarize(topic) → RAG engine (summary mode)
Output: Concise summary with key points
```

### 4. `search(query)` — Fast Retrieval (no generation)
```
Input:  Search query
Action: Returns raw passages (no LLM generation — fast)
Output: "Found 3 relevant passages: ..."
```

### 5. `chat(text)` — Conversational (via Ollama)
```
Input:  Text to respond to
Action: Direct Ollama chat (no RAG)
Output: Friendly conversational response
```

---

## Agentic Behavior Examples

| User says | Gemini thinks | Tools called | User sees |
|---|---|---|---|
| "hello!" | Just chat | None | "Hey! Ready to study?" |
| "what is behavioral finance?" | Needs answer | `ask("behavioral finance")` | Answer from Finance PDF |
| "quiz me on chapter 3" | Needs quiz | `quiz("chapter 3", 5)` | 5 MCQ questions |
| "too easy, make it harder" | Adapts | `quiz("chapter 3 advanced", 5)` | Harder questions |
| "explain that last one" | Follow-up | `ask("explain [previous topic] deeper")` | Deeper explanation |
| "search about loss aversion AND quiz me" | Chains | `search("loss aversion")` + `quiz("loss aversion", 3)` | Passages + quiz |
| "I don't get it" | Clarifies | `ask("explain [topic] simply")` | Simplified explanation |

---

## Files to Create

| File | Purpose |
|---|---|
| `app/plugins/brain/__init__.py` | Package init |
| `app/plugins/brain/handler.py` | BrainPlugin — Gemini agent + tool router |
| `app/llm/gemini.py` | Minimal Gemini API client (httpx, no SDK) |

## Files to Modify

| File | Change |
|---|---|
| `app/config.py` | Add `gemini_api_key` field |
| `.env.example` | Add `GEMINI_API_KEY` placeholder |
| `.env` | Add Gemini API key |
| `app/main.py` | Register BrainPlugin before StudyPlugin |
| `app/bot/dispatcher.py` | Route free text → BrainPlugin (not StudyPlugin) |
| `app/plugins/study/handler.py` | Expose `_ask()`, `_quiz()`, `_summarize()` as class methods |
| `pyproject.toml` | May need no changes (httpx already installed) |
| `PLAN.md` | Update implementation status |

---

## Implementation Steps

### Step 1: Gemini Provider
- Create `app/llm/gemini.py` — minimal httpx client
- Support: system prompt, user message, return text
- Config: model, api_key, timeout

### Step 2: BrainPlugin Handler
- `on_load()` → validate Gemini key
- `handle()` → send to Gemini, parse TOOL: lines, execute tools
- `_process_agent_response()` → find TOOL: lines, execute, replace
- `_execute_tool()` → route to the right handler

### Step 3: Expose StudyPlugin Methods
- Extract `_ask()`, `_quiz()`, `_summarize()` as class/static methods
- These can be called WITHOUT going through `handle()`
- Keep `handle()` for /commands

### Step 4: Wire Up
- Register BrainPlugin in main.py
- Update dispatcher to route free text → BrainPlugin
- Add Gemini config

### Step 5: Test
- "hello" → chat response (no tool) ✅
- "what is X?" → ask() → RAG answer ✅
- "quiz me" → quiz() → questions ✅
- "too hard" → quiz(adjusted) → easier questions ✅
- "tell me more" → ask(follow-up) → deeper answer ✅
