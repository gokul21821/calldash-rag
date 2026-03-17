"""OpenAI embedding generation."""

import logging
from openai import OpenAI

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"


def get_embeddings(text_list: list[str]) -> list[list[float]]:
    """Convert text to vectors using OpenAI's text-embedding-3-small model."""
    if not text_list:
        return []
    response = _client.embeddings.create(
        input=text_list,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]
