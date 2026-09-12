from __future__ import annotations

from agent.api.models import (
    AddDocumentRequest,
    AddDocumentResponse,
)
from rag.ingest import ingest_document


def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
    """Add a document to the existing Chroma knowledge base."""

    collection_name = "fixed_size_collection"

    chunks_added = ingest_document(
        document_name=request.document_name,
        content=request.content,
        collection_name=collection_name,
    )

    return AddDocumentResponse(
        document_name=request.document_name,
        chunks_added=chunks_added,
        collection=collection_name,
        message="Document added to the knowledge base.",
    )