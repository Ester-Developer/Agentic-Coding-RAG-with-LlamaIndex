"""Check stages 2-5: nodes per file, sample nodes, and Pinecone vector count.

Chunking is checked locally (no API calls). Pinecone count needs PINECONE_API_KEY.
Run ingest.py first to populate Pinecone.
"""
import sys
import time
from collections import Counter

from llama_index.core.schema import MetadataMode

from rag.loader import ConfigError
from rag.net import configure_tls
from rag.pipeline import load_and_chunk


def main() -> int:
    try:
        cfg, docs, nodes = load_and_chunk()
        configure_tls(cfg)
    except (FileNotFoundError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Documents: {len(docs)}  ->  Nodes: {len(nodes)}")
    for name, n in Counter(nd.metadata["file_name"] for nd in nodes).items():
        print(f"  {name}: {n} nodes")

    # One sample per file, up to 3
    seen, samples = set(), []
    for nd in nodes:
        if nd.metadata["file_name"] not in seen:
            seen.add(nd.metadata["file_name"])
            samples.append(nd)
    for nd in samples[:3]:
        print("\n" + "=" * 70)
        print(f"id: {nd.id_}   ref_doc_id: {nd.ref_doc_id}")
        for k, v in nd.metadata.items():
            print(f"    {k}: {v}")
        print(f"    excluded from embedding: {nd.excluded_embed_metadata_keys}")
        print("    --- text as sent to the embedding model (first 300 chars) ---")
        print(nd.get_content(metadata_mode=MetadataMode.EMBED)[:300])

    print("\n" + "=" * 70)
    try:
        from rag.vector_store import get_vector_store, vector_count

        vs = get_vector_store(cfg["pinecone"], cfg["embedding"]["dimension"])
        ns = cfg["pinecone"].get("namespace") or ""
        count = vector_count(vs, ns)
        # Pinecone stats are eventually consistent right after an upsert.
        for _ in range(10):
            if count == len(nodes):
                break
            time.sleep(3)
            count = vector_count(vs, ns)
        print(f"Pinecone '{cfg['pinecone']['index_name']}': {count} vectors (expected {len(nodes)})")
    except Exception as e:  # report, don't hide
        print(f"Pinecone check skipped/failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
