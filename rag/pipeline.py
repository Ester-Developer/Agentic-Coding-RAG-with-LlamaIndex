"""Helper for check scripts: load + chunk without any API calls.
The full ingestion now runs as a Workflow: see rag/flows/ingestion_flow.py."""
from __future__ import annotations

from pathlib import Path

from rag.chunking import chunk_documents
from rag.loader import DEFAULT_CONFIG, load_config, load_documents


def load_and_chunk(config_path: str | Path = DEFAULT_CONFIG):
    cfg = load_config(config_path)
    docs = load_documents(config_path)
    nodes = chunk_documents(docs, **cfg["chunking"])
    return cfg, docs, nodes
