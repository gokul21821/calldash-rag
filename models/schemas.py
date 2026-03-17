"""Pydantic request/response models for the RAG API."""

from pydantic import BaseModel


class IngestPayload(BaseModel):
    """Payload for document ingestion webhook."""

    user_id: str
    document_id: str
    file_url: str


class RetrievePayload(BaseModel):
    """Payload for RAG retrieval (voice agent)."""

    user_id: str
    document_ids: list[str]
    user_query: str
    limit: int | None = None  # Default 2; max chunks for voice (300-800 chars target)
