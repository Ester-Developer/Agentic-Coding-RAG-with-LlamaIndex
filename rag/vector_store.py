"""Pinecone index creation/verification and vector store access."""
from __future__ import annotations

from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag.embedding import _require_env


class DimensionMismatchError(Exception):
    pass


def get_pinecone_index(pc_cfg: dict, dimension: int):
    """Create the serverless index if missing; if it exists, verify its dimension and metric."""
    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
    name = pc_cfg["index_name"]
    metric = pc_cfg.get("metric", "cosine")

    if not pc.has_index(name):
        print(f"Creating Pinecone index '{name}' (dim={dimension}, {metric}, {pc_cfg['cloud']}/{pc_cfg['region']})...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=pc_cfg["cloud"], region=pc_cfg["region"]),
        )  # blocks until the index is ready
    else:
        desc = pc.describe_index(name)
        if desc.dimension != dimension:
            raise DimensionMismatchError(
                f"Pinecone index '{name}' has dimension {desc.dimension}, but embedding model "
                f"produces {dimension}. Use a different index_name or delete the index."
            )
        if desc.metric != metric:
            raise DimensionMismatchError(f"Pinecone index '{name}' uses metric '{desc.metric}', expected '{metric}'.")
    return pc.Index(name)


def get_vector_store(pc_cfg: dict, dimension: int) -> PineconeVectorStore:
    return PineconeVectorStore(
        pinecone_index=get_pinecone_index(pc_cfg, dimension),
        namespace=pc_cfg.get("namespace") or None,
    )


def vector_count(vector_store: PineconeVectorStore, namespace: str = "") -> int:
    stats = vector_store._pinecone_index.describe_index_stats()
    if namespace:
        ns = stats.namespaces.get(namespace)
        return ns.vector_count if ns else 0
    return stats.total_vector_count
