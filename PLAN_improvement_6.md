# Magic Grimoire — Improvement Plan #6: Document-Aware Retrieval

## Problem
When a user asks about a specific document, the retriever returns chunks from OTHER documents too:

```
User: "key takeaways from world economy and financial paradigm"

Retriever returns:
  1. Finance for Normal People (score: 0.69)  ← WRONG DOC
  2. World Economy (score: 0.61)               ← Correct but lower!
  3. Finance for Normal People (score: 0.58)  ← WRONG DOC
```

**Why:** nomic-embed-text maps "world economy" and "finance" to similar vectors. The retriever doesn't know which document the user is referring to.

## Solutions

### Solution A: Agent-Pass Document Name (Primary)
Gemini extracts the document name from the user's query and passes it to the tool:

```
User: "key takeaways from world economy"

Gemini → TOOL: ask(query="key takeaways", document="World Economy")
```

The RAG engine boosts chunks from the specified document by 1.5× score:

```
Before boosting:
  1. Finance (0.69)
  2. World Economy (0.61)
  3. Finance (0.58)

After boosting (World Economy × 1.5):
  1. World Economy (0.92) ← boosted!
  2. Finance (0.69)
  3. Finance (0.58)
```

### Solution B: Query Expansion (Complement)
Append the document name to the query before retrieval:

```
Original: "key takeaways from world economy"
Expanded: "key takeaways from world economy [The World Economy and Financial System]"
```

This helps the embedding find more relevant chunks from that document.

## Implementation

### Step 1: Agent Prompt
Add to tool definition:
```
TOOL: ask(query, difficulty="normal", document=None)
  — document: optional document name to focus search on.
  If the user mentions a specific document (e.g., "in the Finance book"),
  extract the document name and pass it here.
```

### Step 2: RAG Engine — Document Boosting
In `query_with_sources()` and `query()`, add document parameter:

```python
async def query_with_sources(self, question, difficulty="normal", document=None):
    retriever = self._get_retriever()
    nodes = retriever.retrieve(question)
    
    if document:
        # Boost scores for chunks from the specified document
        doc_lower = document.lower()
        for node in nodes:
            fname = node.metadata.get("file_name", "").lower()
            if doc_lower in fname or any(
                word in fname for word in doc_lower.split()
            ):
                node.score = (node.score or 0) * 1.5
        
        # Re-sort by boosted score
        nodes.sort(key=lambda n: n.score or 0, reverse=True)
```

### Step 3: Tool Integration
Update `_tool_ask()` in BrainPlugin to accept and pass `document` parameter:

```python
async def _tool_ask(query, difficulty="normal", document=None):
    result = await ask_query(query, difficulty=difficulty, document=document)
    ...
```

### Step 4: StudyPlugin
Update `ask_query()` to accept and forward `document`:

```python
async def ask_query(query, difficulty="normal", document=None):
    result = await _rag_engine.query_with_sources(
        query, difficulty=difficulty, document=document
    )
```

### Files to Change

| File | Change |
|---|---|
| `app/rag/engine.py` | Add `document` parameter to `query_with_sources()`, boost scores for matching docs |
| `app/plugins/study/handler.py` | `ask_query()` passes `document` to engine |
| `app/plugins/brain/handler.py` | `_tool_ask()` accepts `document`, agent prompt updated, `_tool_ask()` passes it through |

### Example Flow

```
User: "what are the key takeaways from the world economy book?"

Gemini: TOOL: ask(
  query="key takeaways", 
  document="World Economy"
)

RAG engine retrieves → boosts "World Economy" chunks × 1.5
  → Re-sorts → top results are from the right document
  → Generates answer from boosted context
  → Returns answer citing "The World Economy and Financial System"
```

### Without document mention
If the user doesn't mention a document, behavior is unchanged — search all documents normally.
