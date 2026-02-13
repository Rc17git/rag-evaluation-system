import time
from rag.retriever import Retriever
from rag.prompt_builder import build_prompt
from rag.generator import Generator



class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def ask_rag(self, query: str, top_k: int = 3):
        start_time = time.time()

        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        prompt = build_prompt(query, retrieved_chunks)

        response = self.generator.generate(prompt)

        latency = time.time() - start_time

        return {
            "mode": "RAG",
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "response": response,
            "latency": latency
        }

    def ask_base(self, query: str):
        start_time = time.time()

        # Base LLM gets NO external context
        prompt = f"""
<|system|>
You are an expert in ML systems optimization.
Answer the question clearly and concisely.
<|end|>

<|user|>
Question:
{query}
<|end|>

<|assistant|>
"""

        response = self.generator.generate(prompt)

        latency = time.time() - start_time

        return {
            "mode": "BASE",
            "query": query,
            "response": response,
            "latency": latency
        }
