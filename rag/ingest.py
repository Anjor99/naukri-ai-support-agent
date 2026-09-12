"""
ingest.py
Ingest a single document into the existing Chroma knowledge base.

Pipeline:
    document content
        ↓
    Document object
        ↓
    chunking
        ↓
    embeddings
        ↓
    existing Chroma collection
"""

from pathlib import Path

from rag.loader import Document
from rag.chunking import fixed_size_chunks
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore


DEFAULT_COLLECTION = "fixed_size_collection"


def ingest_document(
    document_name: str,
    content: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """
    Add a new document to the existing Chroma knowledge base.

    Args:
        document_name: Name of the document, e.g. "new_policy.txt".
        content: Full document content. The first line may optionally be
                 "Document: <description>" to match the existing KB format.
        collection_name: Chroma collection to update.

    Returns:
        Number of chunks added to the collection.

    Raises:
        ValueError: If document name or content is empty.
    """

    if not document_name.strip():
        raise ValueError("document_name cannot be empty")

    if not content.strip():
        raise ValueError("content cannot be empty")

    # Ensure the document has a .txt-style filename.
    filename = Path(document_name).name
    if not filename.lower().endswith(".txt"):
        filename = f"{filename}.txt"

    # Use the filename (without extension) as the stable document ID,
    # matching rag.loader.load_documents().
    doc_id = Path(filename).stem

    # Match loader.py's handling of the first line.
    lines = content.strip().splitlines()

    if lines and lines[0].startswith("Document:"):
        desc = lines[0][len("Document:"):].strip()
        text = "\n".join(lines[1:]).strip()
    else:
        desc = lines[0].strip() if lines else ""
        text = "\n".join(lines).strip()

    if not text:
        raise ValueError(
            "Document must contain text after the optional 'Document:' line."
        )

    # Build the same Document object used by the existing chunking pipeline.
    document = Document(
        doc_id=doc_id,
        filename=filename,
        path="",
        text=text,
        desc=desc,
    )

    # Use the same fixed-size chunking strategy used by the current
    # production RAG collection.
    chunks = fixed_size_chunks(
        [document],
        chunk_size=250,
        overlap=50,
    )

    if not chunks:
        raise ValueError("Document produced no chunks.")

    # Use the same embedding model and persistent ChromaDB configuration.
    embedding_model = EmbeddingModel()
    vector_store = VectorStore(embedding_model)

    # IMPORTANT:
    # get_or_create_collection() connects to the existing persistent
    # Chroma collection instead of creating a separate temporary database.
    collection = vector_store.create_collection(collection_name)

    # Add the chunks + embeddings + metadata to Chroma.
    vector_store.add_chunks(collection, chunks)

    print(
        f"[ingest] Added '{filename}' "
        f"to '{collection_name}' ({len(chunks)} chunks)."
    )

    return len(chunks)


if __name__ == "__main__":
    # Manual test:
    # python -m rag.ingest
    sample_content = """Document: Test Remote Work Policy
Employees may be eligible for remote work depending on their role and business requirements.
Managers review remote-work requests based on team needs, performance, and operational requirements.
"""

    count = ingest_document(
        document_name="test_remote_work.txt",
        content=sample_content,
    )

    print(f"Chunks added: {count}")