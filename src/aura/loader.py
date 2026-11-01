from pathlib import Path


def load_documents(folder):
    """Read every .txt file in a folder and return a list of dicts."""
    documents = []
    for file_path in Path(folder).glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        documents.append({"source": file_path.name, "text": text})
    return documents


def split_into_chunks(text, chunk_size=200):
    """Split text into pieces of roughly chunk_size words."""
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_size):
        chunk = " ".join(words[start:start + chunk_size])
        chunks.append(chunk)
    return chunks