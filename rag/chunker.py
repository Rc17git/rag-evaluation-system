from typing import List, Dict

def chunk_text(document: Dict, chunk_size: int = 400, overlap: int = 50) -> List[Dict]:
    """
    Splits text into overlapping chunks.
    """
    text = document["text"]
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"{document['source']}_{chunk_id}",
            "text": chunk_text,
            "source": document["source"]
        })

        start += chunk_size - overlap
        chunk_id += 1

    return chunks
