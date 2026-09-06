"""
loader.py
Loads .txt documents from the knowledge_base folder into a simple
in-memory structure that downstream chunking/indexing code can use.
"""

import os
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    doc_id: str        # stable id, derived from filename (no extension)
    filename: str       # original filename, e.g. "remote_work.txt"
    path: str           # full path to the file
    text: str           # raw text content
    desc: str           # First line of the text, used as a short description (Text after 'Document:')


def load_documents(folder_path: str = "knowledge_base") -> List[Document]:
    """
    Loads every .txt file in `folder_path` into a Document object.

    Raises:
        FileNotFoundError: if the folder doesn't exist.
        ValueError: if no .txt files are found (fail loud, don't silently
                    proceed with an empty knowledge base).
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Knowledge base folder not found: {folder_path}")

    documents: List[Document] = []

    # sorted() -> deterministic order, makes doc_ids stable across runs
    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".txt"):
            continue

        full_path = os.path.join(folder_path, filename)
        
        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            lines = text.splitlines()

        if not text:
            print(f"[loader] WARNING: {filename} is empty, skipping.")
            continue

        doc_id = os.path.splitext(filename)[0]  # e.g. "remote_work"
        doc_desc = lines[0] if text else ""  # First line as description, also cut off Document: prefix if present
        if doc_desc.startswith("Document:"):
            doc_desc = doc_desc[9:].strip()  # Remove "Document:" prefix and strip whitespace
            
        raw_text = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""  # Exclude the first line from the text
        documents.append(
            Document(doc_id=doc_id, filename=filename, path=full_path, text=raw_text, desc=doc_desc)
        )

    if not documents:
        raise ValueError(
            f"No non-empty .txt files found in '{folder_path}'. "
            "Check the folder path and file extensions."
        )

    print(f"[loader] Loaded {len(documents)} documents from '{folder_path}'.")
    return documents


if __name__ == "__main__":
    # Quick manual sanity check: python rag/loader.py
    docs = load_documents()
    for d in docs:
        # Print everthing, only first few words of the text to avoid flooding the console
        print(f"Document ID: {d.doc_id}, Filename: {d.filename}, Path: {d.path}, Description: {d.desc}, Text (first 100 chars): {d.text[:100]}..., Length: {len(d.text)} characters") 