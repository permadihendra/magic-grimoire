# Magic Grimoire — PLAN: Retrieval Quality + Feedback Learning

## Problem Diagnosis

Log analysis shows the bot retrieves from wrong documents even with 2.0x score boosting:

```
User: "what is the top key takeaway from world economy and financial paradigm books"

Gemini: "I will search through BOTH the 'World Economy' and 'Financial Paradigm' documents..."
        → Treats ONE book as TWO separate documents
        → May not pass document parameter at all, or passes wrong partial name

Retrieval: Finance chunks score 0.80 (high semantic similarity)
           World Economy chunks score 0.40 (lower even with 2x boost → 0.80)
           → TIE! Finance wins tiebreaker → wrong source used
```

### Root Causes (3 layers)

| Layer | Problem | Why |
|---|---|---|
| **Agent** | Gemini doesn't pass `document` reliably | Prompt doesn't emphasize book detection. Gemini splits one book name into "two documents" |
| **Retrieval** | Score boosting can't overcome huge initial score gaps | Finance at 0.80, World Economy at 0.40 → even 2x = 0.80 tie |
| **Context** | Ollama LLM sees mixed context | If Finance chunks dominate, answer comes from wrong book |

---

## Three-Layer Solution

### Layer 1: Force Named-Doc Chunks into Context (Post-Retrieval Filtering)

**Current:** Boost scores, hope the named doc rises to the top.

**Better:** GUARANTEE chunks from the named document appear in context:

```python
def retrieve_with_document_filter(question, document=None):
    nodes = retriever.retrieve(question)  # top-k from ALL docs
    
    if document:
        # SEPARATE into two pools
        doc_nodes = [n for n in nodes if _doc_matches(n, document)]
        other_nodes = [n for n in nodes if not _doc_matches(n, document)]
        
        # ALWAYS include top-2 from named document, then fill with others
        selected = doc_nodes[:2] + other_nodes[:(5 - len(doc_nodes[:2]))]
        selected = selected[:5]
    else:
        selected = nodes[:5]
    
    return selected
```

**Result:** Even if World Economy chunks score 0.30, at least 2 appear in context. The LLM will see them alongside Finance chunks and can answer from the right book.

### Layer 2: Feedback Learning (Simple RL)

**The idea:** When user says "wrong source" or "that's from Finance not World Economy", the bot learns and applies a penalty for that session.

```
User: "key takeaways from world economy"
Bot:  [answer from Finance] ← WRONG
User: "that's from the wrong book, it's from Finance for Normal People"
Bot:  [detects feedback] → "Got it, I'll exclude Finance for Normal People. Let me try again with just World Economy."
Bot:  [retrieves again, this time EXCLUDING Finance] → [correct answer]
```

**Implementation:**

```python
# Per-chat learning state
class FeedbackLearner:
    def __init__(self):
        self.penalized_docs: dict[str, float] = {}  # filename → penalty
        self.boosted_docs: dict[str, float] = {}     # filename → boost
    
    def apply_penalty(self, filename: str):
        self.penalized_docs[filename] = -0.5  # 50% score reduction
    
    def apply_boost(self, filename: str):
        self.boosted_docs[filename] = 1.5      # 50% score boost
    
    def adjust_score(self, filename: str, base_score: float) -> float:
        score = base_score
        for doc, penalty in self.penalized_docs.items():
            if _doc_matches_name(filename, doc):
                score *= (1.0 + penalty)  # e.g., 0.8 * 0.5 = 0.4
        for doc, boost in self.boosted_docs.items():
            if _doc_matches_name(filename, doc):
                score *= (1.0 + boost)   # e.g., 0.3 * 2.5 = 0.75
        return score
```

**Gemini detects feedback via new tool:**

```
TOOL: feedback(type="wrong_source", detail="answer came from Finance for Normal People instead of World Economy")
TOOL: feedback(type="good", detail="perfect answer from World Economy")
```

### Layer 3: Hybrid Search (BM25 + Vector)

**The idea:** Combine keyword matching with semantic matching. If user says "world economy", BM25 finds chunks with exact phrase matches, which the World Economy book will naturally have more of.

```python
# Simple hybrid: keyword pre-filter before vector search
def keyword_prefilter(question, index):
    """Boost chunks that contain query keywords."""
    keywords = set(question.lower().split()) - STOPWORDS
    
    # Retrieve more chunks initially
    nodes = retriever.retrieve(question, similarity_top_k=20)
    
    # Boost chunks containing keywords
    for node in nodes:
        text_lower = node.text.lower()
        keyword_matches = sum(1 for kw in keywords if kw in text_lower)
        if keyword_matches > 0:
            node.score = (node.score or 0) * (1.0 + keyword_matches * 0.2)
    
    # Re-sort
    nodes.sort(key=lambda n: n.score or 0, reverse=True)
    return nodes[:5]
```

This doesn't require a full BM25 index — just a simple keyword presence check that boosts chunks containing the user's exact words.

---

## How The Feedback Loop Works (Full Flow)

```
Turn 1:
  User: "what are key takeaways from world economy?"
  Gemini: TOOL: ask(query="key takeaways", document="World Economy")
  Engine: retrieve → Finance (0.80), World Economy (0.40 boosted to 0.80)
          → Filter: World Economy chunks forced into context (Layer 1)
          → Generate answer
  Bot: [answer with mixed sources — mostly World Economy but some Finance]

Turn 2 (feedback):
  User: "that answer is pulling from Finance for Normal People, not World Economy"
  Gemini: 
    "You're right, let me exclude Finance for Normal People and search again."
    TOOL: feedback(type="wrong_source", detail="Finance for Normal People")
    TOOL: ask(query="key takeaways", document="World Economy")
  
  Engine: → FeedbackLearner penalizes Finance (-0.5)
          → Finance now: 0.80 * 0.5 = 0.40
          → World Economy: 0.40 boosted to 0.80 (document boost) + forced inclusion
          → Context: ONLY World Economy chunks
          → Answer: "The key takeaway from World Economy is..."
  
  Bot: "Here's the corrected answer from World Economy: ..."

Turn 3 (positive feedback):
  User: "perfect, that's what I wanted"
  Gemini:
    "Great! I'll remember that." 
    TOOL: feedback(type="good", detail="World Economy answer was correct")
  
  Engine: → FeedbackLearner boosts World Economy (+0.5)
          → Future queries about similar topics will prefer World Economy
```

---

## What Gemini Needs to Detect

New agent prompt rules:

```
FEEDBACK DETECTION:
- If user says "wrong source", "wrong book", "that's from [X] not [Y]", 
  "you looked at the wrong document":
  → Use TOOL: feedback(type="wrong_source", detail="[document name]")
  → Then retry with corrected parameters

- If user says "good", "perfect", "exactly", "that's right":
  → Use TOOL: feedback(type="good", detail="[document name]")
  → This helps me learn which sources are relevant for which topics

- When retrying after feedback:
  → ALWAYS pass the corrected document name
  → If user says "from X not Y", pass document="X"
```

---

## Implementation Plan

### Phase 1: Post-Retrieval Filtering (Layer 1)
- **1a.** In `engine.py`, modify `query_with_sources()` and `retrieve_only()`: when `document` is specified, guarantee top-2 chunks from that doc in context
- **1b.** Improve `_doc_matches()` to handle partial names, underscores, stopwords
- **Files:** `engine.py` only (~30 lines)

### Phase 2: Feedback Learning (Layer 2)
- **2a.** New `app/rag/feedback.py` — `FeedbackLearner` class
- **2b.** Store per-chat learners in a simple dict (keyed by chat_id), clear on bot restart
- **2c.** Add `TOOL: feedback(type, detail)` to BrainPlugin + agent prompt
- **2d.** Wire `FeedbackLearner` into `query_with_sources()` 
- **2e.** Improve Gemini prompt: detect wrong-source feedback, retry with corrected params
- **Files:** `feedback.py` (new), `engine.py`, `brain/handler.py`, agent prompt

### Phase 3: Hybrid Search (Layer 3)
- **3a.** Add keyword pre-filtering in `engine.py` — boost chunks with query keywords
- **Files:** `engine.py` only (~20 lines)

### Phase 4: Better Agent Prompting
- **4a.** Update agent prompt: emphasize detecting book names, passing `document` parameter
- **4b.** Use `/files` output to show Gemini what documents are available
- **4c.** When user mentions a partial name, Gemini should match against available docs
- **Files:** `brain/handler.py` (prompt update)

---

## Files Summary

| Phase | File | Change |
|---|---|---|
| 1 | `app/rag/engine.py` | Post-retrieval filtering: force named-doc chunks into context |
| 2 | `app/rag/feedback.py` | **NEW** — `FeedbackLearner` class |
| 2 | `app/rag/engine.py` | Apply feedback penalties/boosts to scores |
| 2 | `app/plugins/brain/handler.py` | Add `_tool_feedback()`, update agent prompt |
| 3 | `app/rag/engine.py` | Keyword pre-filtering |
| 4 | `app/plugins/brain/handler.py` | Better agent prompt for document detection |

---

## Why Not Full Reinforcement Learning?

True RL (policy gradients, reward functions, model fine-tuning) is:
- Too heavy for an 8GB GPU (qwen2.5:7b already takes 4.7 GB)
- Requires thousands of training examples — we have single-digit feedback
- Overkill for "which document to search" — it's a routing problem, not a generation problem

**Our approach ("Simple RL"):**
- Per-session, per-document penalty/boost scores
- Learns from 1-2 feedback examples per session
- No model training — just score adjustment
- Clears on restart (acceptable — sessions are short)

This is effectively a **contextual bandit** — one action (which doc to prefer), immediate reward (user feedback), simple policy (boost/penalty).

---

## Future: LLM Reranker (Optional Phase 5)

If the above isn't enough, add a lightweight reranker:

```
After retrieval:
  1. Get top-10 chunks
  2. Ask a small Ollama model (qwen2.5:3b, 1.9 GB): 
     "Which of these chunks is most relevant to: [query]? Output scores 1-10."
  3. Re-sort by LLM scores
  4. Use top-3 for generation
```

Cost: +2-3s latency, +1.9 GB VRAM. Only for challenging queries.
