import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Get project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VECTOR_PATH = os.path.join(BASE_DIR, "vector_store", "faiss.index")
META_PATH = os.path.join(BASE_DIR, "vector_store", "metadata.npy")


class Retriever:
    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("intfloat/e5-small-v2")

        print("Loading FAISS index...")
        self.index = faiss.read_index(VECTOR_PATH)

        print("Loading metadata...")
        self.metadata = np.load(META_PATH, allow_pickle=True)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in indices[0]:
            results.append(self.metadata[i])

        return results
