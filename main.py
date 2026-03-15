import os
import logging
import requests
import io
import uuid
import fitz  # PyMuPDF
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Distance, VectorParams, PayloadSchemaType
from openai import OpenAI
import logging
from dotenv import load_dotenv

# Setup Logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Clients - Ensure these Env Vars are set!
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL", "https://localhost:6333"), 
    api_key=os.getenv("QDRANT_API_KEY"),
)

COLLECTION_NAME = "voice_rag_index"

# --- Schema Definitions ---
class IngestPayload(BaseModel):
    user_id: str
    document_id: str
    file_url: str

class RetrievePayload(BaseModel):
    user_id: str
    document_ids: list[str]
    user_query: str

# --- Helper Functions ---
def get_embeddings(text_list):
    """Converts text to vectors using OpenAI's latest small model."""
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-small"
    )
    return [e.embedding for e in response.data]

def process_document(data: IngestPayload):
    try:
        logger.info(f"🚀 Starting ingestion for {data.document_id}")
        
        # A. Download
        response = requests.get(data.file_url, timeout=10)
        response.raise_for_status()
        
        # B. Extract Text
        with fitz.open(stream=io.BytesIO(response.content), filetype="pdf") as doc:
            # We join lines with spaces to fix the "line-break" issue in PDFs
            full_text = ""
            for page in doc:
                full_text += page.get_text("text") + " "
        
        # Clean up whitespace
        full_text = " ".join(full_text.split())
        
        if len(full_text) < 20:
            logger.error("❌ Extraction failed or document too short.")
            return

        # C. RECURSIVE CHUNKING (The Fix)
        # We split the text into chunks of ~800 characters with a 100-character overlap
        # This prevents losing context at the edges of chunks.
        chunk_size = 800
        overlap = 100
        chunks = []
        
        for i in range(0, len(full_text), chunk_size - overlap):
            chunk = full_text[i : i + chunk_size]
            chunks.append(chunk)

        logger.info(f"📦 Successfully split into {len(chunks)} chunks")

        # D. Embed & Upsert (All chunks)
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
                    "chunk_index": i
                }
            })
        
        # Upsert in batches of 100 to avoid hitting API payload limits
        for i in range(0, len(points), 100):
            qdrant.upsert(
                collection_name=COLLECTION_NAME, 
                points=points[i:i+100]
            )
            
        logger.info(f"✅ Upserted {len(points)} points to Qdrant.")
        return "Success"

    except Exception as e:
        logger.error(f"💥 Ingestion Error: {str(e)}")
        return str(e)

# --- Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Ensure the Qdrant collection exists on startup."""
    try:
        collections = qdrant.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            logger.info(f"Creating collection: {COLLECTION_NAME}")
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            qdrant.create_payload_index(COLLECTION_NAME, "user_id", PayloadSchemaType.KEYWORD)
            qdrant.create_payload_index(COLLECTION_NAME, "document_id", PayloadSchemaType.KEYWORD)
    except Exception as e:
        logger.error(f"Qdrant startup error: {e}")

@app.post("/ingest")
async def ingest(payload: IngestPayload, bg_tasks: BackgroundTasks):
    # This immediately tells Supabase "Got it!", then processes the PDF in the background
    bg_tasks.add_task(process_document, payload)
    return {"status": "Processing in background"}

@app.post("/retrieve")
async def retrieve(payload: RetrievePayload):
    try:
        logger.info(f"🔍 Searching for: {payload.user_query}")
        
        # The modern Qdrant query method
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=get_embeddings([payload.user_query])[0], # Pass the vector here
            query_filter=Filter(
                must=[
                    FieldCondition(key="user_id", match=MatchValue(value=payload.user_id)),
                    FieldCondition(key="document_id", match=MatchAny(any=payload.document_ids))
                ]
            ),
            limit=3
        )
        
        # In the new API, search_result.points contains the hits
        results = [hit.payload["text"] for hit in search_result.points if hit.payload]
        
        if not results:
            return {"context": "No relevant info found."}
            
        return {"context": "\n\n".join(results)}
        
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {str(e)}")
        # If query_points fails, try the legacy .search as a fallback
        try:
            logger.info("Retrying with legacy search...")
            legacy_result = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=get_embeddings([payload.user_query])[0],
                query_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=payload.user_id)),
                        FieldCondition(key="document_id", match=MatchAny(any=payload.document_ids))
                    ]
                ),
                limit=3
            )
            results = [hit.payload["text"] for hit in legacy_result]
            return {"context": "\n\n".join(results)}
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Final error: {str(e2)}")