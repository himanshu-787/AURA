import json
import re
import sys
import time

from loader import load_documents, split_into_chunks
from embedding_search import build_index, semantic_search

CHUNK_SIZE = 50
TOP_K = 3
IDK = "i don't know based on the provided documents"


def build_chunks(chunk_size):
    chunks = []
    for doc in load_documents("data"):
        for piece in split_into_chunks(doc["text"], chunk_size=chunk_size):
            chunks.append({"source": doc["source"], "text": piece})
    return chunks


def retrieval_hit(item, results):
    """True if a retrieved chunk is from the right file and has a keyword."""
    for score, chunk in results:
        text = chunk["text"].lower()
        if chunk["source"] == item["expected_source"] and any(
            k.lower() in text for k in item["keywords"]
        ):
            return True
    return False


def answer_ok(item, reply):
    """Check the format of the answer (not whether it is truly correct)."""
    text = reply.lower().replace("\u2019", "'")
    said_idk = IDK in text
    if item["answerable"]:
        has_citation = re.search(r"[\[【]\s*\d+\s*[\]】]", reply) is not None
        return (not said_idk) and has_citation
    return said_idk


def main(retrieval_only):
    with open("tests/eval_set.json", encoding="utf-8") as f:
        items = json.load(f)

    chunks = build_chunks(CHUNK_SIZE)
    index = build_index(chunks)
    print(f"{len(chunks)} chunks | chunk_size={CHUNK_SIZE} | top_k={TOP_K}\n")

    if not retrieval_only:
        from rag import answer

    hits, answerable_count, answers_ok = 0, 0, 0
    for number, item in enumerate(items, start=1):
        results = semantic_search(item["question"], chunks, index, top_k=TOP_K)
        line = f"Q{number}: {item['question'][:60]}"

        if item["answerable"]:
            answerable_count += 1
            hit = retrieval_hit(item, results)
            hits += hit
            line += f" | retrieval: {'HIT' if hit else 'MISS'}"

        if not retrieval_only:
            reply, _ = answer(item["question"], chunks, index, top_k=TOP_K)
            ok = answer_ok(item, reply)
            answers_ok += ok
            line += f" | answer: {'OK' if ok else 'BAD'}"
            time.sleep(1)  # be gentle with free-tier rate limits

        print(line)

    print(f"\nRetrieval hit rate@{TOP_K}: {hits}/{answerable_count}")
    if not retrieval_only:
        print(f"Answer format OK: {answers_ok}/{len(items)}")


if __name__ == "__main__":
    main(retrieval_only="--retrieval-only" in sys.argv)