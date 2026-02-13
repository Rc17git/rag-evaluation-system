from pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline()

    query = "How can CPU inference latency be reduced?"

    print("\n========== BASE LLM ==========\n")
    base_result = rag.ask_base(query)
    print(base_result["response"])
    print(f"\nLatency: {base_result['latency']:.2f} seconds")

    print("\n========== RAG LLM ==========\n")
    rag_result = rag.ask_rag(query, top_k=3)
    print(rag_result["response"])
    print(f"\nLatency: {rag_result['latency']:.2f} seconds")
