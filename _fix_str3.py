import sys

path = "/home/hendra/my-projects/magic-grimoire/app/rag/engine.py"
with open(path) as f:
    data = f.read()

# Fix the broken fallback prompt — it has literal \n instead of \\n in the f-string
old_broken = """            fallback_context = "\\n\\n".join(chunk_texts[:2])
            fallback_prompt = (
                f"Based ONLY on this context, answer the question.\\n\\n"
                f"Context: {fallback_context}\\n\\n"
                f"Question: {question}\\n\\n"
                f"Answer:"
            )"""

new_fixed = """            fallback_context = "\\n\\n".join(chunk_texts[:2])
            fallback_prompt = (
                f"Based ONLY on this context, answer the question.\\n\\n"
                f"Context: {fallback_context}\\n\\n"
                f"Question: {question}\\n\\n"
                f"Answer:"
            )"""

if old_broken in data:
    data = data.replace(old_broken, new_fixed, 1)
    with open(path, 'w') as f:
        f.write(data)
    print("Fixed!")
else:
    # Check what's actually there
    idx = data.find("fallback_context")
    if idx >= 0:
        print("Found at", idx)
        print(repr(data[idx:idx+300]))
    sys.exit(1)