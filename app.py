"""Gradio UI — Agentic Coding Knowledge Base. Each question runs the Query Workflow."""
import html
import logging
from collections import Counter

import gradio as gr

from rag.flows.query_flow import FALLBACK_NOTICE, QueryResult, QueryWorkflow
from rag.loader import PROJECT_ROOT, load_config
from rag.net import configure_tls
from rag.structured.repository import StructuredRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for noisy in ("httpx", "httpx2", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

CONFIG = load_config()
configure_tls(CONFIG)
_workflow = QueryWorkflow(config=CONFIG, timeout=180)   # reused: keeps the Pinecone/Cohere clients

TOOL_LABELS = {"claude_code": "Claude Code", "kiro": "Kiro"}
TYPE_LABELS = {"decisions": "Decisions", "rules": "Rules & guidelines",
               "warnings": "Sensitivities & limitations", "changes": "Changes"}
ROUTES = {
    "semantic": ("route-semantic", "Semantic search"),
    "structured": ("route-structured", "Structured query"),
    "fallback": ("route-fallback", "Structured → semantic fallback"),
}
# UI chrome is English; the workflow's notices are keyed by fallback reason
NOTICE_EN = {
    FALLBACK_NOTICE["invalid_structured_query"]: "Could not build a valid structured query, so semantic search was used instead.",
    FALLBACK_NOTICE["no_structured_results"]: "No structured items matched the question, so semantic search was used instead.",
}

EXAMPLES = [
    "What was decided about Kafka?",
    "Which login API fields changed?",
    "Which component is marked as sensitive?",
    "Which technical limitation repeats across docs?",
    "What is the latest guideline on proto files?",
    "How does login flow from Frontend to Gateway?",
    "What are the code review rules?",
]


def render_details(result: QueryResult) -> str:
    """Second message under the answer: route badge, optional notice, source cards (HTML).
    Kept apart from the answer so that the answer's copy button copies clean text."""
    css_class, label = ROUTES.get(result.route, ("route-semantic", result.route))
    parts = [f'<div class="route-badge {css_class}">{label}</div>']
    if result.notice:
        parts.append(f'<div class="notice">{html.escape(NOTICE_EN.get(result.notice, result.notice))}</div>')
    if result.sources:
        cards = "".join(
            f'<div class="source"><span class="src-num">[{s.number}]</span>'
            f'<span class="src-tool tool-{html.escape(s.tool)}">{TOOL_LABELS.get(s.tool, s.tool)}</span>'
            f'<span class="src-path">{html.escape(s.relative_path)}</span>'
            f'<span class="src-header">{html.escape(s.header)}</span>'
            + (f'<span class="src-date">{html.escape(s.date)}</span>' if s.date else "") + "</div>"
            for s in result.sources)
        parts.append(f'<div class="sources"><div class="sources-title">Sources</div>{cards}</div>')
    return "\n".join(parts)


async def respond(message: str, history: list):
    """Show the question at once, then the answer (plain markdown) + details (route, sources)."""
    message = (message or "").strip()
    if not message:
        yield history, ""
        return
    history = history + [{"role": "user", "content": message}]
    yield history + [{"role": "assistant", "content": "Searching the knowledge base…"}], ""
    result = await _workflow.run(query=message)
    yield history + [{"role": "assistant", "content": result.answer},
                     {"role": "assistant", "content": render_details(result)}], ""


async def answer(message: str, history=None) -> str:
    """Programmatic helper (scripts/tests): answer text + details as one string."""
    result = await _workflow.run(query=message)
    return result.answer + "\n\n" + render_details(result)


# after an answer arrives, scroll so the latest question is at the top -> the answer is read from its start
# Scrolls ONLY the chat's own scroll box (scrollIntoView would also scroll the whole page down).
SCROLL_TO_ANSWER_JS = """() => {
  const pageY = window.scrollY;
  setTimeout(() => {
    const qs = document.querySelectorAll('#chat [data-testid="user"]');
    if (!qs.length) return;
    const msg = qs[qs.length - 1];
    let box = msg.parentElement;
    while (box && box !== document.body) {
      const oy = getComputedStyle(box).overflowY;
      if ((oy === 'auto' || oy === 'scroll') && box.scrollHeight > box.clientHeight) break;
      box = box.parentElement;
    }
    if (box && box !== document.body) {
      const top = msg.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop - 8;
      box.scrollTo({top: top, behavior: 'smooth'});
    }
    window.scrollTo({top: pageY});
  }, 150);
}"""


def knowledge_panel() -> str:
    """Static summary of what the knowledge layer contains (files per tool + extracted items)."""
    files = Counter(d["tool"] for d in CONFIG["documents"])
    items = StructuredRepository(PROJECT_ROOT / CONFIG["structured"]["store_path"]).items()
    by_type = Counter(i.item_type for i in items)
    tools = "".join(f'<div class="stat"><span class="tool-dot tool-{t}"></span>{TOOL_LABELS.get(t, t)}'
                    f'<b>{n} {"file" if n == 1 else "files"}</b></div>' for t, n in files.items())
    types = "".join(f'<div class="stat">{TYPE_LABELS[t]}<b>{by_type.get(t, 0)}</b></div>' for t in TYPE_LABELS)
    if not items:
        types = '<div class="stat">Not extracted yet — run <code>python extract.py</code></div>'
    return f"""
<div class="panel">
  <div class="panel-title">Knowledge base</div>
  <div class="panel-sub">Sources</div>
  {tools}
  <div class="panel-sub">Extracted items</div>
  {types}
  <div class="panel-title" style="margin-top:20px">How questions are answered</div>
  <div class="legend"><span class="route-badge route-semantic">Semantic</span><span>How / why / what is</span></div>
  <div class="legend"><span class="route-badge route-structured">Structured</span><span>Lists, latest, time ranges, changes</span></div>
  <div class="legend"><span class="route-badge route-fallback">Fallback</span><span>No matching structured items</span></div>
  <div class="panel-note">Answers are based only on the indexed files and always cite their sources.
  If the files don't contain the answer, the system says so.</div>
</div>"""


HEADER = """
<div class="hero">
  <div class="hero-title">Agentic Coding Knowledge Base</div>
  <div class="hero-sub">The markdown files written by the project's AI coding tools (Claude Code, Kiro), unified into one
  queryable knowledge layer — see what the tools rely on when they make decisions, and get new team members up to speed.</div>
</div>"""

CSS = """
.gradio-container { max-width: 98% !important; width: 98% !important; margin: auto; }
.hero { background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%); color: #fff;
        border-radius: 14px; padding: 14px 22px; margin-bottom: 2px; box-shadow: 0 8px 24px rgba(79,70,229,.2); }
.hero-title { font-size: 1.35rem; font-weight: 800; letter-spacing: -.015em; }
.hero-sub { font-size: .9rem; opacity: .92; margin-top: 4px; line-height: 1.45; }
/* example chips: small, wrap, one or two lines */
.chips-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; margin: 2px 2px 0; }
#chips { gap: 6px !important; flex-wrap: nowrap !important; overflow-x: auto; scrollbar-width: thin; padding-bottom: 2px; }
#chips button { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; font-size: .82rem !important;
                padding: 4px 12px !important; border-radius: 999px !important; white-space: nowrap !important; }
.panel { background: var(--block-background-fill); border: 1px solid var(--border-color-primary);
         border-radius: 16px; padding: 18px; }
.panel-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }
.panel-sub { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6; margin: 12px 0 6px; }
.stat { display: flex; align-items: center; gap: 8px; padding: 7px 10px; border-radius: 10px;
        background: var(--background-fill-secondary); margin-bottom: 6px; font-size: .92rem; }
.stat b { margin-left: auto; color: #6d28d9; }
.tool-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.tool-dot.tool-claude_code, .src-tool.tool-claude_code { background: #d97706; }
.tool-dot.tool-kiro, .src-tool.tool-kiro { background: #0891b2; }
.legend { font-size: .86rem; margin: 7px 0; display: flex; align-items: center; gap: 10px; }
.legend .route-badge { min-width: 86px; text-align: center; }
.panel-note { font-size: .82rem; opacity: .72; margin-top: 14px; line-height: 1.5; }
.route-badge { display: inline-block; font-size: .78rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; }
.route-semantic { background: #dbeafe; color: #1e40af; }
.route-structured { background: #ede9fe; color: #5b21b6; }
.route-fallback { background: #fef3c7; color: #92400e; }
.notice { margin-top: 8px; padding: 8px 12px; border-radius: 10px; background: #fff7ed; color: #9a3412;
          border-left: 4px solid #f97316; font-size: .9rem; }
.answer-text { margin-top: 6px; line-height: 1.7; unicode-bidi: plaintext; text-align: start; }
.answer-text p, .answer-text li, #chat p, #chat li, #chat ol, #chat ul { unicode-bidi: plaintext; text-align: start; }
.sources { margin-top: 12px; border-top: 1px dashed var(--border-color-primary); padding-top: 10px; }
.sources-title { font-size: .75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; opacity: .6; margin-bottom: 6px; }
.source { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: .82rem; padding: 6px 9px;
          border-radius: 8px; background: var(--background-fill-secondary); margin-bottom: 5px; }
.src-num { font-weight: 800; color: #6d28d9; }
.src-tool { color: #fff; padding: 1px 8px; border-radius: 999px; font-weight: 700; font-size: .74rem; }
.src-path { font-family: ui-monospace, monospace; }
.src-header { opacity: .8; }
.src-date { margin-left: auto; opacity: .6; }
/* chat: keep only the per-message copy icon (hide the clear/trash button) */
#chat button[aria-label*="lear" i], #chat button[title*="lear" i],
#chat button[aria-label*="elete" i], #chat button[title*="elete" i] { display: none !important; }
/* make the input area obvious */
#ask-box label span, #ask-box span[data-testid="block-info"] { font-weight: 700; font-size: .95rem; color: #5b21b6; }
#ask-box textarea { font-size: 1.05rem !important; min-height: 56px !important; unicode-bidi: plaintext; text-align: start; }
#ask-box { border: 2px solid #7c3aed !important; border-radius: 14px !important;
           box-shadow: 0 0 0 4px rgba(124,58,237,.12) !important; background: var(--block-background-fill) !important; }
#ask-box:focus-within { box-shadow: 0 0 0 5px rgba(124,58,237,.25) !important; }
"""

THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="pink", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks(title="Agentic Coding Knowledge Base", fill_width=True) as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            # Plain Chatbot (not ChatInterface). buttons=["copy"] = copy icon on every message; the clear/trash
            # button is hidden in CSS. No retry/undo/delete/edit/like.
            chatbot = gr.Chatbot(
                elem_id="chat", height="calc(100vh - 330px)", min_height=260, show_label=False,
                sanitize_html=False, buttons=["copy"], feedback_options=None, autoscroll=False,
                placeholder="<b>Ask anything about the decisions, rules, sensitivities and changes "
                            "documented by the coding tools.</b><br>Type your question in the box below, "
                            "or pick an example under it.")
            textbox = gr.Textbox(placeholder="Type your question here and press Enter…", show_label=True,
                                 label="Ask a question (Hebrew or English)", elem_id="ask-box", submit_btn="Ask")
            textbox.submit(respond, [textbox, chatbot], [chatbot, textbox]).then(None, js=SCROLL_TO_ANSWER_JS)
            with gr.Row(elem_id="chips"):
                for q in EXAMPLES:
                    gr.Button(q, size="sm", variant="secondary").click(lambda q=q: q, outputs=textbox)
        with gr.Column(scale=1, min_width=300):
            gr.HTML(knowledge_panel())

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
