"""Run the Extraction Workflow. Unchanged files are skipped; pass --force to re-extract everything."""
import asyncio
import logging
import sys

from rag.flows.extraction_flow import run_extraction
from rag.loader import load_config
from rag.net import configure_tls


def main() -> int:
    configure_tls(load_config())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summary = asyncio.run(run_extraction(force="--force" in sys.argv))
    if not summary.ok:
        print(f"ERROR: {summary.error}", file=sys.stderr)
        return 1
    print("\n=== Extraction summary ===")
    print(f"extracted files: {summary.extracted_files or '-'}")
    print(f"unchanged files (skipped): {summary.unchanged_files or '-'}")
    print(f"sections sent to LLM: {summary.sections}   duplicates merged: {summary.duplicates_merged}")
    print(f"items by type (whole store): {summary.items_by_type}")
    print(f"items by tool (whole store): {summary.items_by_tool}")
    print(f"files to retry next run (had failed sections): {summary.failed_files or '-'}")
    print(f"skipped ({len(summary.skipped)}):")
    for s in summary.skipped:
        print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
