from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="User's support query.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Existing conversation ID for persisted memory.",
    )


class AskResponse(BaseModel):
    conversation_id: str
    route: str
    response: str


class AddDocumentRequest(BaseModel):
    document_name: str = Field(
        ...,
        min_length=1,
        description="Name of the document to add to the knowledge base.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="Full text content of the document.",
    )


class AddDocumentResponse(BaseModel):
    document_name: str
    chunks_added: int
    collection: str
    message: str