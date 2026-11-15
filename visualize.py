"""Write HTML diagrams of all workflows (every step and event)."""
from pathlib import Path

from llama_index.utils.workflow import draw_all_possible_flows

from rag.flows.extraction_flow import ExtractionWorkflow
from rag.flows.ingestion_flow import IngestionWorkflow
from rag.flows.query_flow import QueryWorkflow
from rag.loader import load_config

if __name__ == "__main__":
    Path("docs").mkdir(exist_ok=True)
    flows = {
        "docs/workflow_ingestion.html": IngestionWorkflow(),
        "docs/workflow_extraction.html": ExtractionWorkflow(),
        "docs/workflow_query.html": QueryWorkflow(config=load_config()),
    }
    for filename, wf in flows.items():
        draw_all_possible_flows(wf, filename=filename)
        print(f"Wrote {filename}")
