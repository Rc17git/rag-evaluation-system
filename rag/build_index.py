import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.data_loader import load_documents
from rag.chunker import chunk_text


DATA_PATH = "../data"
INDEX_PATH = "../vector_store/faiss.index"
META_PATH = "../vector_store/metadata.npy"

def build_faiss_index():

    print("Loading embedding model...")
    model = SentenceTransformer("intfloat/e5-small-v2")

    print("Loading documents...")
    documents = load_documents(DATA_PATH)

    print("Chunking documents...")
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)

    print(f"Total chunks: {len(all_chunks)}")

    texts = [chunk["text"] for chunk in all_chunks]

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, INDEX_PATH)

    print("Saving metadata...")
    np.save(META_PATH, all_chunks)

    print("Indexing complete.")

if __name__ == "__main__":
    build_faiss_index()
