import os
from typing import List, Dict

def load_documents(data_path: str) -> List[Dict]:
    """
    Loads all .txt files from the data directory.
    Returns list of dicts with text and metadata.
    """
    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            file_path = os.path.join(data_path, file)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "text": text,
                "source": file,
            })

    return documents
