from fastapi import FastAPI
from pydantic import BaseModel
import time

from rag.pipeline import RAGPipeline

app = FastAPI(title="RAG Evaluation API")

# Load pipeline ONCE at startup
rag_pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str
    mode: str = "rag"   # "rag" or "base"
    top_k: int = 3


@app.get("/")
def root():
    return {"message": "RAG Evaluation API is running."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    start_time = time.time()

    if request.mode.lower() == "base":
        result = rag_pipeline.ask_base(request.query)
        retrieved_sources = []
    else:
        result = rag_pipeline.ask_rag(request.query, top_k=request.top_k)
        retrieved_sources = [
            chunk["source"] for chunk in result["retrieved_chunks"]
        ]

    total_time = time.time() - start_time

    return {
        "mode": result["mode"],
        "query": request.query,
        "response": result["response"],
        "latency": result["latency"],
        "total_request_time": total_time,
        "retrieved_sources": retrieved_sources
    }