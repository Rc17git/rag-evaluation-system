import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer("intfloat/e5-small-v2")


def semantic_similarity(text1: str, text2: str):
    emb1 = embedding_model.encode([text1])
    emb2 = embedding_model.encode([text2])

    return cosine_similarity(emb1, emb2)[0][0]


def token_overlap_ratio(answer: str, context: str):
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    overlap = answer_tokens.intersection(context_tokens)

    if len(answer_tokens) == 0:
        return 0.0

    return len(overlap) / len(answer_tokens)


def novel_token_ratio(answer: str, context: str):
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context.lower().split())

    novel = answer_tokens - context_tokens

    if len(answer_tokens) == 0:
        return 0.0

    return len(novel) / len(answer_tokens)
