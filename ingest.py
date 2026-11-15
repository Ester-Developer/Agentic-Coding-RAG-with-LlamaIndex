"""Run the Ingestion Workflow (safe to re-run: no duplicates)."""
import asyncio
import logging
import sys

from rag.flows.ingestion_flow import run_ingestion
from rag.loader import load_config
from rag.net import configure_tls


def main() -> int:
    configure_tls(load_config())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = asyncio.run(run_ingestion())
    if not summary.ok:
        print(f"ERROR: {summary.error}", file=sys.stderr)
        return 1
    print(f"Ingested {summary.documents} documents -> {summary.nodes} nodes.")
    for s in summary.skipped:
        print(f"  skipped: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
