"""Qdrant client, collection management, and vector search."""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    VectorParams,
    PayloadSchemaType,
)

from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)


def ensure_collection_exists() -> None:
    """Create the voice_rag_index collection if it does not exist."""
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            qdrant_client.create_payload_index(COLLECTION_NAME, "user_id", PayloadSchemaType.KEYWORD)
            qdrant_client.create_payload_index(COLLECTION_NAME, "document_id", PayloadSchemaType.KEYWORD)
    except Exception as e:
        logger.error(f"Qdrant startup error: {e}")
        raise


def delete_document_points(user_id: str, document_id: str) -> None:
    """Delete all points for a given user_id and document_id (tenant-safe, idempotent re-ingestion)."""
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            ]
        ),
    )


def upsert_points(points: list[dict]) -> None:
    """Upsert points to the collection in batches of 100."""
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=batch)


def search_points(
    query_vector: list[float],
    user_id: str,
    document_ids: list[str],
    limit: int = 3,
) -> list[str]:
    """
    Search for relevant chunks filtered by user_id and document_ids.
    Returns list of text payloads from top matches.
    """
    try:
        result = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="document_id", match=MatchAny(any=document_ids)),
                ]
            ),
            limit=limit,
        )
        return [hit.payload["text"] for hit in result.points if hit.payload and "text" in hit.payload]
    except Exception:
        # Fallback to legacy search API
        logger.info("Retrying with legacy search...")
        legacy_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    FieldCondition(key="document_id", match=MatchAny(any=document_ids)),
                ]
            ),
            limit=limit,
        )
        return [hit.payload["text"] for hit in legacy_result if hit.payload and "text" in hit.payload]
