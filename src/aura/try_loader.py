from loader import load_documents, split_into_chunks

docs = load_documents("data")
print("Documents found:", len(docs))

for doc in docs:
    chunks = split_into_chunks(doc["text"], chunk_size=20)
    print(doc["source"], "->", len(chunks), "chunks")
    print("First chunk:", chunks[0])