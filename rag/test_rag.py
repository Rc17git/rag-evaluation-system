from pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline()

    query = "How can CPU inference latency be reduced?"

    retrieved = rag.retriever.retrieve(query)

    print("\nRETRIEVED CHUNKS:\n")
    for i, chunk in enumerate(retrieved):
        print(f"\nChunk {i+1}:\n{chunk['text']}\n")

    response = rag.ask(query)

    print("\nFINAL RESPONSE:\n")
    print(response)
