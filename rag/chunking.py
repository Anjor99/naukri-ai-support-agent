"""
chunking.py
Contains functions to split text into chunks of a specified size, with optional overlap.
Fixed size chunking and Sentence-based chunking are supported.
"""

import re
from dataclasses import dataclass
import enum
from typing import List

from rag.loader import Document


class ChunkingStrategy(enum.Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE_BASED = "sentence_based"


@dataclass
class Chunk:
    chunk_id: str  # stable id, derived from doc_id and chunk index
    doc_id: str    # id of the document this chunk belongs to
    text: str      # text content of the chunk
    strategy: ChunkingStrategy


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size chunking with overlap
# ---------------------------------------------------------------------------

def fixed_size_chunks(
    documents: List[Document],
    chunk_size: int = 250,
    overlap: int = 50,
) -> List[Chunk]:
    """
    Splits each document's text into fixed-size character windows with
    overlap between consecutive windows.

    Args:
        chunk_size: number of characters per chunk.
        overlap: number of characters shared between consecutive chunks.
                 Must be < chunk_size.

    Returns:
        List[Chunk] with strategy=ChunkingStrategy.FIXED_SIZE.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer")

    chunks: List[Chunk] = []
    step = chunk_size - overlap

    for doc in documents:
        text = doc.text
        n = len(text)
        idx = 0

        if n <= chunk_size:
            # Document shorter than one window -> single chunk, no loop needed
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_fixed_{idx}",
                doc_id=doc.doc_id,
                text=text,
                strategy=ChunkingStrategy.FIXED_SIZE,
            ))
            continue

        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_fixed_{idx}",
                doc_id=doc.doc_id,
                text=text[start:end],
                strategy=ChunkingStrategy.FIXED_SIZE,
            ))
            idx += 1
            if end == n:
                break
            start += step

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-based chunking
# ---------------------------------------------------------------------------

# Lightweight sentence splitter (no external download / nltk punkt required).
# Splits on '.', '!', '?' followed by whitespace, while trying to avoid
# breaking on common abbreviations. Good enough for short policy-style docs;
# swap for nltk.sent_tokenize if you want a more robust splitter.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def sentence_based_chunks(
    documents: List[Document],
    sentences_per_chunk: int = 2,
    sentence_overlap: int = 0,
) -> List[Chunk]:
    """
    Splits each document into sentences, then groups N sentences per chunk.

    Args:
        sentences_per_chunk: how many sentences to group into one chunk.
        sentence_overlap: how many trailing sentences from the previous
                           chunk to repeat at the start of the next
                           (0 = no overlap, pure sequential grouping).

    Returns:
        List[Chunk] with strategy=ChunkingStrategy.SENTENCE_BASED.
    """
    if sentence_overlap >= sentences_per_chunk:
        raise ValueError("sentence_overlap must be smaller than sentences_per_chunk")
    if sentences_per_chunk <= 0:
        raise ValueError("sentences_per_chunk must be a positive integer")
    if sentence_overlap < 0:
        raise ValueError("sentence_overlap must be a non-negative integer")

    chunks: List[Chunk] = []
    step = sentences_per_chunk - sentence_overlap

    for doc in documents:
        sentences = _split_sentences(doc.text)
        n = len(sentences)
        if n == 0:
            continue

        idx = 0
        start = 0
        while start < n:
            end = min(start + sentences_per_chunk, n)
            chunk_text = " ".join(sentences[start:end])
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_sent_{idx}",
                doc_id=doc.doc_id,
                text=chunk_text,
                strategy=ChunkingStrategy.SENTENCE_BASED,
            ))
            idx += 1
            if end == n:
                break
            start += step

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher (optional convenience)
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    strategy: ChunkingStrategy,
    **kwargs,
) -> List[Chunk]:
    """
    Convenience dispatcher so callers can pick a strategy via the enum
    instead of calling fixed_size_chunks / sentence_based_chunks directly.
    """
    if strategy == ChunkingStrategy.FIXED_SIZE:
        return fixed_size_chunks(documents, **kwargs)
    elif strategy == ChunkingStrategy.SENTENCE_BASED:
        return sentence_based_chunks(documents, **kwargs)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


if __name__ == "__main__":
    # Quick manual sanity check: python rag/chunking.py
    from rag.loader import load_documents

    docs = load_documents("knowledge_base")

    fixed = fixed_size_chunks(docs)
    sent = sentence_based_chunks(docs, sentences_per_chunk=2)

    print(f"\nFixed-size chunks:     {len(fixed)}")
    print(f"Sentence-based chunks: {len(sent)}")

    print("\n--- Sample fixed chunk ---")
    print(fixed[0])
    print("\n--- Sample sentence chunk ---")
    print(sent[0])