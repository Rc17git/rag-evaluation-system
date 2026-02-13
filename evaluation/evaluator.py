import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from rag.pipeline import RAGPipeline
from evaluation.metrics import semantic_similarity, token_overlap_ratio, novel_token_ratio

class Evaluator:
    def __init__(self):
        self.pipeline = RAGPipeline()

    def evaluate_query(self, query: str):
        base = self.pipeline.ask_base(query)
        rag = self.pipeline.ask_rag(query)

        context_text = ""
        for chunk in rag["retrieved_chunks"]:
            context_text += chunk["text"] + " "

        base_similarity = semantic_similarity(base["response"], context_text)
        rag_similarity = semantic_similarity(rag["response"], context_text)

        base_overlap = token_overlap_ratio(base["response"], context_text)
        rag_overlap = token_overlap_ratio(rag["response"], context_text)

        base_novel = novel_token_ratio(base["response"], context_text)
        rag_novel = novel_token_ratio(rag["response"], context_text)

        return {
            "query": query,
            "base_latency": base["latency"],
            "rag_latency": rag["latency"],
            "base_similarity": base_similarity,
            "rag_similarity": rag_similarity,
            "base_overlap": base_overlap,
            "rag_overlap": rag_overlap,
            "base_novel_ratio": base_novel,
            "rag_novel_ratio": rag_novel
        }
