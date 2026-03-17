"""
One-time Qdrant migration script.

Recreates the voice_rag_index collection with:
- on_disk=True (original vectors on disk)
- scalar quantization INT8 (quantized vectors in RAM for fast search)
- payload indexes on user_id and document_id

Usage (from project root):
    python -m scripts.migrate_vectors

Requires: QDRANT_URL, QDRANT_API_KEY (if using Qdrant Cloud) in .env
"""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from qdrant_client import QdrantClient
from qdrant_client import models

from config import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME


def main() -> None:
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
    )

    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}' with on_disk + INT8 quantization...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE,
            on_disk=True,
        ),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            )
        ),
    )

    print("Creating payload indexes...")
    client.create_payload_index(COLLECTION_NAME, "user_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(COLLECTION_NAME, "document_id", models.PayloadSchemaType.KEYWORD)

    print("Migration complete. Collection is ready for ingestion.")


if __name__ == "__main__":
    main()
