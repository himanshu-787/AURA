import numpy as np
from sentence_transformers import SentenceTransformer

from loader import load_documents, split_into_chunks
from simple_search import search as keyword_search

model = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks):
    """Turn every chunk into a vector. This is our 'index'."""
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, normalize_embeddings=True)


def semantic_search(question, chunks, index, top_k=3):
    question_vector = model.encode([question], normalize_embeddings=True)[0]
    scores = index @ question_vector
    best = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), chunks[i]) for i in best]


if __name__ == "__main__":
    all_chunks = []
    for doc in load_documents("data"):
        for piece in split_into_chunks(doc["text"], chunk_size=50):
            all_chunks.append({"source": doc["source"], "text": piece})

    index = build_index(all_chunks)
    print("Index shape:", index.shape)

    question = input("Question: ")

    print("\n=== KEYWORD (baseline) ===")
    for score, chunk in keyword_search(question, all_chunks, top_k=2):
        print(f"{score:.3f} | {chunk['source']} | {chunk['text'][:100]}...")

    print("\n=== SEMANTIC (embeddings) ===")
    for score, chunk in semantic_search(question, all_chunks, index, top_k=2):
        print(f"{score:.3f} | {chunk['source']} | {chunk['text'][:100]}...")