from evaluation.evaluator import Evaluator
import pandas as pd

queries = [
    "How does quantization improve CPU inference performance?",
    "What trade-offs exist between batch size and latency?",
    "How does CPU profiling guide optimization?",
    "Why is operator fusion useful?",
    "What metrics should be monitored in deployment?",
]

if __name__ == "__main__":
    evaluator = Evaluator()

    results = []

    for query in queries:
        print(f"\nEvaluating: {query}")
        result = evaluator.evaluate_query(query)
        results.append(result)

    df = pd.DataFrame(results)
    print("\nFinal Results:\n")
    print(df)

    df.to_csv("evaluation_results.csv", index=False)
