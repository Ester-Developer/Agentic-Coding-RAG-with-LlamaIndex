"""Stage 2: split loaded Documents into Nodes.

MarkdownNodeParser cuts by md headers; SentenceSplitter then only splits
sections still longer than chunk_size (shorter ones pass through unchanged).
"""
from __future__ import annotations

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode

# Metadata that carries meaning -> included in the text sent to the embedding model.
# Everything else (paths, dates, sizes) stays as metadata for filtering only.
EMBED_METADATA_KEYS = {"tool", "doc_type", "file_name", "header_path", "header"}

# Not useful for an LLM reading the chunk later.
LLM_EXCLUDED_KEYS = ["file_path", "file_size", "file_type", "creation_date", "last_modified_ts"]


def _date_to_int(date_str: str | None) -> int | None:
    """'2026-10-25' -> 20261025 (numeric, so Pinecone can filter with $gte/$lte)."""
    return int(date_str.replace("-", "")) if date_str else None


def _section_header(text: str) -> str:
    """MarkdownNodeParser puts the section's own header on the first line."""
    first = text.lstrip().split("\n", 1)[0]
    return first.lstrip("#").strip() if first.startswith("#") else ""


def chunk_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[BaseNode]:
    # Stable document ids, so node ids and ref_doc_id are stable across runs.
    for doc in documents:
        doc.id_ = doc.metadata["relative_path"]

    sections = MarkdownNodeParser().get_nodes_from_documents(documents)

    for node in sections:
        md = node.metadata
        md["header"] = _section_header(node.get_content())
        md["last_modified_ts"] = _date_to_int(md.get("last_modified_date"))
        # Set before splitting so SentenceSplitter accounts for metadata length and children inherit it.
        node.excluded_embed_metadata_keys = [k for k in md if k not in EMBED_METADATA_KEYS]
        node.excluded_llm_metadata_keys = list(LLM_EXCLUDED_KEYS)

    nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents(sections)

    # Deterministic ids: "<relative_path>::<chunk#>" -> re-ingest overwrites instead of duplicating.
    counters: dict[str, int] = {}
    for node in nodes:
        doc_id = node.metadata["relative_path"]
        i = counters.get(doc_id, 0)
        counters[doc_id] = i + 1
        node.id_ = f"{doc_id}::{i:04d}"
    return nodes
