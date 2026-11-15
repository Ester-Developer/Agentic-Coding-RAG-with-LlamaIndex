"""Semantic route (stage B): each step alone + every routing path end-to-end."""
from llama_index.core.workflow import Context

from rag.flows.query_flow import (NOT_ENOUGH_INFO, NOT_ENOUGH_INFO_EN, QUERY_TOO_SHORT_EN, AnswerDrafted, ContextInsufficient,
                                  ContextSufficient, QueryStart, QueryState, QueryWorkflow, RephraseRequested,
                                  RetrievalDone, RouteRequested, SearchRequested, SemanticSearchRequested)
from rag.structured.repository import StructuredRepository
from tests.conftest import SEMANTIC, TODAY, FakeIndex, FakeLLM, FakeSelector, make_node


def wf(cfg, results=(), texts=(), tmp_path=None, **kw):
    return QueryWorkflow(config=cfg, index=FakeIndex(results), llm=FakeLLM(texts=texts),
                         selector=FakeSelector(SEMANTIC), today=TODAY, timeout=30, **kw)


GOOD = [make_node("a", 0.8), make_node("b", 0.7, header="Architecture")]
LOW = [make_node("c", 0.1)]


# ---------- single steps ----------
async def test_validate_query_rejects_empty(cfg):
    w = wf(cfg)
    out = await w.validate_query(Context(w), QueryStart(query="  "))
    assert out.result.stop_reason == "invalid_query"


async def test_validate_query_accepts(cfg):
    w = wf(cfg)
    out = await w.validate_query(Context(w), QueryStart(query="how to run tests?"))
    assert isinstance(out, RouteRequested)


async def test_retrieve_merges_into_state(cfg):
    w = wf(cfg, results=[GOOD])
    ctx = Context(w)
    out = await w.retrieve(ctx, SemanticSearchRequested(query_text="q", top_k=5))
    state = await ctx.store.get_state()
    assert isinstance(out, RetrievalDone) and state.attempt == 1 and set(state.collected_nodes) == {"a", "b"}


async def test_evaluate_sufficient_and_insufficient(cfg):
    w = wf(cfg)
    ctx = Context(w)
    async with ctx.store.edit_state() as s:
        s.collected_nodes = {n.node.node_id: n for n in GOOD}
    assert isinstance(await w.evaluate_retrieval(ctx, RetrievalDone(nodes=GOOD)), ContextSufficient)
    ctx2 = Context(w)
    async with ctx2.store.edit_state() as s:
        s.collected_nodes = {n.node.node_id: n for n in LOW}
    out = await w.evaluate_retrieval(ctx2, RetrievalDone(nodes=LOW))
    assert isinstance(out, ContextInsufficient) and out.reason == "too_few"


async def test_plan_retry_order_and_exhaustion(cfg):
    w = wf(cfg)
    ctx = Context(w)
    async with ctx.store.edit_state() as s:
        s.attempt, s.original_query = 1, "q"
    assert isinstance(await w.plan_retry(ctx, ContextInsufficient(reason="low_score")), RephraseRequested)
    async with ctx.store.edit_state() as s:
        s.attempt = 2
    out = await w.plan_retry(ctx, ContextInsufficient(reason="low_score"))
    assert isinstance(out, SearchRequested) and out.top_k == cfg["query"]["expanded_top_k"]
    async with ctx.store.edit_state() as s:
        s.attempt = 3
    out = await w.plan_retry(ctx, ContextInsufficient(reason="low_score"))
    assert out.result.stop_reason == "attempts_exhausted"


async def test_rephrase_rejects_repeat(cfg):
    w = wf(cfg, texts=["q"])
    ctx = Context(w)
    async with ctx.store.edit_state() as s:
        s.original_query, s.tried_queries = "q", ["q"]
    assert isinstance(await w.rephrase_query(ctx, RephraseRequested()), ContextInsufficient)


async def test_validate_answer_requires_citation(cfg):
    w = wf(cfg)
    out = await w.validate_answer(Context(w), AnswerDrafted(answer="no citations here", nodes=GOOD))
    assert out.result.stop_reason == "not_grounded" and out.result.answer == NOT_ENOUGH_INFO_EN


# ---------- routing paths, end-to-end ----------
async def test_path_success(cfg):
    w = wf(cfg, results=[GOOD], texts=["answer [1][2]"])
    r = await w.run(query="how do I run tests?")
    assert r.grounded and r.route == "semantic" and [s.number for s in r.sources] == [1, 2]


async def test_path_retry_then_success(cfg):
    w = wf(cfg, results=[LOW, GOOD], texts=["rephrased query", "answer [1]"])
    r = await w.run(query="how do I run tests?")
    assert r.grounded and w._index.calls[1] == ("rephrased query", cfg["query"]["top_k"])


async def test_path_attempts_exhausted(cfg):
    w = wf(cfg, results=[LOW, LOW, LOW], texts=["other query"])
    r = await w.run(query="something not in docs")
    assert r.stop_reason == "attempts_exhausted" and r.answer == NOT_ENOUGH_INFO_EN   # English question
    assert len(w._index.calls) == cfg["query"]["max_attempts"]


async def test_path_empty_input_never_searches(cfg):
    w = wf(cfg)
    r = await w.run(query="")
    assert r.answer == QUERY_TOO_SHORT_EN and w._index.calls == [] and w._selector.questions == []


async def test_hebrew_question_gets_hebrew_instruction(cfg):
    w = wf(cfg, results=[GOOD], texts=["תשובה [1]"])
    await w.run(query="איך מריצים בדיקות?")
    assert "Answer in Hebrew" in w._llm.prompts[-1]


async def test_no_answer_with_citations_is_not_shown_as_grounded(cfg):
    # regression: "the info doesn't say who developed me [1][2][3][4][5]" was shown as a sourced answer
    w = wf(cfg, results=[GOOD], texts=["NO_ANSWER [1][2]"])
    r = await w.run(query="מי פיתח אותך?")
    assert r.stop_reason == "not_grounded" and r.sources == [] and r.answer == NOT_ENOUGH_INFO


async def test_canned_message_follows_question_language(cfg):
    he = await wf(cfg, results=[[], [], []], texts=["שאילתה אחרת"]).run(query="מה המתכון לעוגה?")
    en = await wf(cfg, results=[[], [], []], texts=["other query"]).run(query="What is the cake recipe?")
    assert he.answer == NOT_ENOUGH_INFO and en.answer == NOT_ENOUGH_INFO_EN
