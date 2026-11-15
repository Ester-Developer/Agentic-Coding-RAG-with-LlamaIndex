"""Stage 1: load source markdown files into LlamaIndex Documents.

Usage from later stages:
    from rag.loader import load_documents
    docs = load_documents()
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from llama_index.core import Document, SimpleDirectoryReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


class ConfigError(Exception):
    """Invalid config or missing source file."""


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # The source project location is machine-specific: RAG_SOURCE_ROOT (e.g. in .env) overrides config.
    load_dotenv(PROJECT_ROOT / ".env")
    if os.getenv("RAG_SOURCE_ROOT"):
        config["source_root"] = os.environ["RAG_SOURCE_ROOT"]

    if "source_root" not in config or not config.get("documents"):
        raise ConfigError(f"{config_path}: 'source_root' and 'documents' are required")

    allowed_tools = set(config.get("tools") or [])
    allowed_types = set(config.get("doc_types") or [])
    for i, entry in enumerate(config["documents"]):
        for key in ("path", "tool", "doc_type"):
            if not entry.get(key):
                raise ConfigError(f"documents[{i}] is missing '{key}': {entry}")
        if allowed_tools and entry["tool"] not in allowed_tools:
            raise ConfigError(f"documents[{i}]: unknown tool '{entry['tool']}' (allowed: {sorted(allowed_tools)})")
        if allowed_types and entry["doc_type"] not in allowed_types:
            raise ConfigError(f"documents[{i}]: unknown doc_type '{entry['doc_type']}' (allowed: {sorted(allowed_types)})")
    return config


def resolve_sources(config: dict) -> dict[str, dict]:
    """Return {absolute_path: custom_metadata}; raise if any file is missing."""
    root = Path(config["source_root"])
    sources: dict[str, dict] = {}
    missing: list[str] = []
    for entry in config["documents"]:
        abs_path = (root / entry["path"]).resolve()
        if not abs_path.is_file():
            missing.append(str(abs_path))
            continue
        sources[str(abs_path)] = {
            "tool": entry["tool"],
            "doc_type": entry["doc_type"],
            "relative_path": Path(entry["path"]).as_posix(),
        }
    if missing:
        raise FileNotFoundError("Source file(s) not found:\n  " + "\n  ".join(missing))
    return sources


def load_documents(config_path: str | Path = DEFAULT_CONFIG) -> list[Document]:
    config = load_config(config_path)
    sources = resolve_sources(config)

    def file_metadata(file_path: str) -> dict:
        # Keep LlamaIndex's default metadata, then add ours on top.
        from llama_index.core.readers.file.base import default_file_metadata_func

        meta = default_file_metadata_func(file_path)
        meta.update(sources[str(Path(file_path).resolve())])
        return meta

    reader = SimpleDirectoryReader(
        input_files=list(sources.keys()),
        file_metadata=file_metadata,
    )
    return reader.load_data()
