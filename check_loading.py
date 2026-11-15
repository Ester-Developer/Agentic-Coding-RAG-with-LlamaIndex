"""Stage 1 check: print files vs Documents, and each Document's metadata + preview."""
import sys
from collections import Counter

from rag.loader import ConfigError, load_config, load_documents


def main() -> int:
    try:
        n_files = len(load_config()["documents"])
        docs = load_documents()
    except (FileNotFoundError, ConfigError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"Files in config: {n_files}")
    print(f"Documents loaded: {len(docs)}")
    per_file = Counter(d.metadata.get("file_name") for d in docs)
    if len(docs) != n_files:
        print("NOTE: reader split files into multiple Documents:", dict(per_file))
    else:
        print("1 Document per file (no split by headers).")

    for i, d in enumerate(docs, 1):
        print("\n" + "=" * 70)
        print(f"[{i}] {d.metadata.get('file_name')}")
        for k, v in d.metadata.items():
            print(f"    {k}: {v}")
        print("    --- first 200 chars ---")
        print(d.text[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
