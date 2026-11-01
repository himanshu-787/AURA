import math
import re
from collections import Counter

from loader import load_documents, split_into_chunks


def tokenize(text):
    """Lowercase text and return a list of words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def to_vector(text):
    """Turn text into word counts, e.g. {'cat': 2, 'sat': 1}."""
    return Counter(tokenize(text))


def cosine_similarity(a, b):
    common_words = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common_words)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(question, chunks, top_k=3):
    question_vector = to_vector(question)
    scored = []
    for chunk in chunks:
        score = cosine_similarity(question_vector, to_vector(chunk["text"]))
        scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    all_chunks = []
    for doc in load_documents("data"):
        for piece in split_into_chunks(doc["text"], chunk_size=50):
            all_chunks.append({"source": doc["source"], "text": piece})

    question = input("Question: ")
    for score, chunk in search(question, all_chunks):
        print(f"\nScore: {score:.3f} | Source: {chunk['source']}")
        print(chunk["text"])