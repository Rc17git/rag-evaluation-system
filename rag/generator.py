from llama_cpp import Llama
import os

# Get project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "tinyllama.gguf")

class Generator:
    def __init__(self):
        print("Loading TinyLlama GGUF with llama.cpp...")

        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,      # Adjust if needed
            n_gpu_layers=1    # Enables Metal acceleration on M1
        )

    def generate(self, prompt: str, max_new_tokens: int = 200):
        output = self.llm(
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop=["</s>"]
        )

        return output["choices"][0]["text"].strip()
