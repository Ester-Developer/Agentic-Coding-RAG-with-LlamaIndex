"""Stage 3: Cohere embedding models (one for documents, one for queries)."""
from __future__ import annotations

import os

from dotenv import load_dotenv
from llama_index.embeddings.cohere import CohereEmbedding

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Missing {name}. Copy .env.example to .env and set it.")
    return value


def get_embed_model(embedding_cfg: dict, for_queries: bool = False) -> CohereEmbedding:
    """Separate instances are required: CohereEmbedding applies a fixed input_type
    to *every* call when one is set, so a single instance can't serve both roles."""
    input_type = embedding_cfg["query_input_type"] if for_queries else embedding_cfg["document_input_type"]
    return CohereEmbedding(
        api_key=_require_env("COHERE_API_KEY"),
        model_name=embedding_cfg["model_name"],
        input_type=input_type,
        embed_batch_size=embedding_cfg.get("embed_batch_size", 48),
    )
