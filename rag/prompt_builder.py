def build_prompt(query: str, retrieved_chunks):
    context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])

    prompt = f"""
<|system|>
You must answer strictly using only the provided context.
Do not add external knowledge.
If information is not present, say:
"I don't have enough information."

When answering, directly reference techniques explicitly mentioned in the context.
<|end|>

<|user|>
Context:
{context}

Question:
{query}
<|end|>

<|assistant|>
"""

    return prompt
