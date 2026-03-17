"""PDF download, text extraction, chunking, and ingestion orchestration."""

import io
import logging
import uuid
import requests
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.schemas import IngestPayload
from services.openai_service import get_embeddings
from services.qdrant_service import delete_document_points, upsert_points
from config import COLLECTION_NAME

logger = logging.getLogger(__name__)


def process_document(data: IngestPayload) -> str | None:
    """
    Download PDF, extract text, chunk, embed, and upsert to Qdrant.
    Returns "Success" on success, or error string on failure.
    """
    try:
        logger.info(f"🚀 Starting ingestion for {data.document_id}")

        # A. Download
        response = requests.get(data.file_url, timeout=10)
        response.raise_for_status()

        # B. Extract Text (preserve newlines for paragraph-aware chunking)
        with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            raw_text = ""
            for page in doc:
                raw_text += page.get_text("text") + "\n"

        # Preserve vertical structure, only trim trailing horizontal whitespace
        full_text = "\n".join(line.rstrip() for line in raw_text.splitlines())

        if len(full_text) < 20:
            logger.error("Extraction failed or document too short.")
            return None

        # C. Chunk with RecursiveCharacterTextSplitter (paragraph/sentence boundaries)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(full_text)

        logger.info(f"📦 Successfully split into {len(chunks)} chunks")

        # D. Delete existing chunks for this document (tenant-safe, idempotent)
        delete_document_points(data.user_id, data.document_id)

        # E. Embed & Upsert
        vectors = get_embeddings(chunks)
        points = []
        for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {
                    "user_id": data.user_id,
                    "document_id": data.document_id,
                    "text": chunk_text,
                    "chunk_index": i,
                },
            })

        upsert_points(points)
        logger.info(f"✅ Upserted {len(points)} points to Qdrant.")
        return "Success"

    except Exception as e:
        logger.error(f"💥 Ingestion Error: {str(e)}")
        return str(e)
