import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(BASE_DIR, "evaluation_results.csv")

df = pd.read_csv(csv_path)

# Compute averages
avg_base_similarity = df["base_similarity"].mean()
avg_rag_similarity = df["rag_similarity"].mean()

avg_base_novel = df["base_novel_ratio"].mean()
avg_rag_novel = df["rag_novel_ratio"].mean()

avg_base_latency = df["base_latency"].mean()
avg_rag_latency = df["rag_latency"].mean()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Similarity
axes[0].bar(["Base", "RAG"], [avg_base_similarity, avg_rag_similarity])
axes[0].set_title("Semantic Similarity (Higher = Better)")
axes[0].set_ylim(0, 1)

# Novel Token Ratio
axes[1].bar(["Base", "RAG"], [avg_base_novel, avg_rag_novel])
axes[1].set_title("Novel Token Ratio (Lower = Better)")
axes[1].set_ylim(0, 1)

# Latency
axes[2].bar(["Base", "RAG"], [avg_base_latency, avg_rag_latency])
axes[2].set_title("Latency (Seconds)")

plt.tight_layout()
plt.show()
