"""Run the router eval: does route_query's selector pick the expected route?
Uses the same select_route() + route descriptions as the Query Workflow (real Cohere call).
--full also runs each question through the whole Query Workflow and prints the final route + answer start."""
import asyncio
import sys
from pathlib import Path

import yaml

from rag.flows.query_flow import QueryWorkflow, load_route_choices, select_route
from rag.loader import load_config
from rag.net import configure_tls

EVAL_FILE = Path(__file__).parent / "eval" / "router_eval.yaml"


async def main(full: bool) -> int:
    cfg = load_config()
    configure_tls(cfg)
    wf = QueryWorkflow(config=cfg, timeout=180)
    selector, choices = wf._get_selector(), load_route_choices()
    questions = yaml.safe_load(EVAL_FILE.read_text(encoding="utf-8"))["questions"]
    correct = 0
    for q in questions:
        route, reason = await select_route(selector, choices, q["question"])
        ok = route == q["expected"]
        correct += ok
        print(f"{'OK  ' if ok else 'MISS'} [{q['kind']:8}] expected={q['expected']:10} got={route:10} | {q['question']}")
        print(f"      reason: {reason[:160]}")
        if full:
            r = await wf.run(query=q["question"])
            print(f"      full run -> route={r.route} stop={r.stop_reason} sources={len(r.sources)}"
                  f"{' notice=' + r.notice if r.notice else ''}")
            print(f"      answer: {r.answer[:200].replace(chr(10), ' ')}")
    print(f"\nRouter accuracy: {correct}/{len(questions)}")
    return 0 if correct == len(questions) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main("--full" in sys.argv)))
