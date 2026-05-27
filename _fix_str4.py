import sys

path = "/home/hendra/my-projects/magic-grimoire/app/rag/engine.py"
with open(path) as f:
    data = f.read()

# The issue: literal \n (actual newlines) inside a Python string literal
# Fix: find the block by unique surrounding context and replace entirely
old_broken_block = """logger.error("[%s]  [p2] TreeSummarize timed out after 30s — fallback to simple prompt", _qid)
            # Fallback: simple join of top 2 chunks only
            fallback_context = "\\n\\n".join(chunk_texts[:2])
            fallback_prompt = (
                f"Based ONLY on this context, answer the question.\n\\n"
                f"Context: {fallback_context}\n\\n"
                f"Question: {question}\n\\n"
                f"Answer:"
            )"""

new_fixed_block = """logger.error("[%s]  [p2] TreeSummarize timed out after 30s — fallback to simple prompt", _qid)
            # Fallback: simple join of top 2 chunks only
            fallback_context = "\\n\\n".join(chunk_texts[:2])
            fallback_prompt = (
                "Based ONLY on this context, answer the question.\\n\\n"
                f"Context: {fallback_context}\\n\\n"
                f"Question: {question}\\n\\n"
                "Answer:"
            )"""

if old_broken_block not in data:
    # Try to find the fallback_context line and see what's there
    idx = data.find("fallback_context = ")
    if idx >= 0:
        print("Found fallback_context at", idx)
        print("Around it:", repr(data[idx:idx+400]))
    sys.exit(1)

data = data.replace(old_broken_block, new_fixed_block, 1)
with open(path, 'w') as f:
    f.write(data)
print("Fixed!")