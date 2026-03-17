"""Environment variables and application settings."""

import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "voice_rag_index"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API Key for /ingest and /retrieve (Supabase team)
API_KEY = os.getenv("API_KEY", os.getenv("INGEST_API_KEY"))
