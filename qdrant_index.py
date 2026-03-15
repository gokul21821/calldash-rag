import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("QDRANT_ENDPOINT") or "http://localhost:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
)
COLLECTION = "voice_rag_index"

# Create collection for OpenAI-sized vectors (1536 dims)
if client.collection_exists(COLLECTION):
    print(f"Collection '{COLLECTION}' already exists. Skipping creation.")
else:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    # Create mandatory payload indexes
    client.create_payload_index(COLLECTION, "user_id", PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION, "document_id", PayloadSchemaType.KEYWORD)

print("Qdrant ready.")