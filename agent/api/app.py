from __future__ import annotations

from fastapi import FastAPI

from agent.api.document import add_document
from agent.api.ask import ask_agent
from agent.api.models import (
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    AskResponse,
)


app = FastAPI(
    title="Naukri.com Support Agent",
    version="1.0.0",
    description="FastAPI deployment for the Naukri.com Recruitment & HR support agent.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return ask_agent(request)


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document_endpoint(
    request: AddDocumentRequest,
) -> AddDocumentResponse:
    return add_document(request)
