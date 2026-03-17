"""FastAPI app entry point and route wiring."""

import re
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException
from fastapi.security import APIKeyHeader

from config import API_KEY
from models.schemas import IngestPayload, RetrievePayload
from services.document_service import process_document
from services.openai_service import get_embeddings
from services.qdrant_service import ensure_collection_exists, search_points

load_dotenv()


def sanitize_for_voice(text: str) -> str:
    """Clean retrieved text for TTS: remove bullets, collapse newlines, trim whitespace."""
    if not text:
        return ""
    # Replace bullet symbols with dash (TTS-friendly)
    text = re.sub(r"[●○•▪▸►\u200b\u200c\u200d\ufeff]", " - ", text)
    # Collapse multiple newlines to at most 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """Validate x-api-key header. Reject if missing or invalid."""
    if not API_KEY:
        logger.warning("API_KEY not set; accepting all requests (dev mode)")
        return api_key or ""
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key


@app.on_event("startup")
async def startup_event() -> None:
    """Ensure the Qdrant collection exists on startup."""
    ensure_collection_exists()


@app.post("/ingest")
async def ingest(
    payload: IngestPayload,
    bg_tasks: BackgroundTasks,
    _: str = Depends(verify_api_key),
) -> dict:
    """Accept document ingestion webhook; process PDF in background."""
    bg_tasks.add_task(process_document, payload)
    return {"status": "Processing in background"}


@app.post("/retrieve")
async def retrieve(
    payload: RetrievePayload,
    _: str = Depends(verify_api_key),
) -> dict:
    """Return relevant context for RAG from voice agent query."""
    try:
        if not payload.document_ids:
            return {"context": "No documents selected for search."}

        logger.info(f"Searching for: {payload.user_query}")

        query_vector = get_embeddings([payload.user_query])[0]
        limit = payload.limit or 2
        results = search_points(
            query_vector=query_vector,
            user_id=payload.user_id,
            document_ids=payload.document_ids,
            limit=limit,
        )

        if not results:
            return {"context": "No relevant info found."}

        sanitized = [sanitize_for_voice(r) for r in results]
        return {"context": "\n\n".join(sanitized)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
