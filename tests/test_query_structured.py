"""Router + structured route: each step alone and every path (success, invalid query, no results -> fallback)."""
from llama_index.core.workflow import Context

from rag.flows.query_flow import (FallbackRequested, QueryWorkflow, RouteRequested, SemanticSearchRequested,
                                  StructuredQueryBuilt, StructuredQueryInvalid, StructuredQueryRequested,
                                  StructuredQueryValid, StructuredResultsFound)
from rag.structured.schema import StructuredQuery
from tests.conftest import SEMANTIC, STRUCTURED, TODAY, FakeIndex, FakeLLM, FakeSelector, make_node


def wf(cfg, repo, route=STRUCTURED, structured=(), texts=(), results=()):
    return QueryWorkflow(config=cfg, index=FakeIndex(results), llm=FakeLLM(texts=texts, structured=structured),
                         selector=FakeSelector(route), repository=repo, today=TODAY, timeout=30)


# ---------- router ----------
async def test_router_semantic(cfg, repo):
    w = wf(cfg, repo, route=SEMANTIC)
    out = await w.route_query(Context(w), RouteRequested(query="how does login work?"))
    assert isinstance(out, SemanticSearchRequested)


async def test_router_structured(cfg, repo):
    w = wf(cfg, repo, route=STRUCTURED)
    ctx = Context(w)
    out = await w.route_query(ctx, RouteRequested(query="list all decisions"))
    assert isinstance(out, StructuredQueryRequested) and (await ctx.store.get_state()).route == "structured"


async def test_router_failure_defaults_to_semantic(cfg, repo):
    class Broken:
        async def aselect(self, *a):
            raise RuntimeError("boom")
    w = QueryWorkflow(config=cfg, selector=Broken(), repository=repo, today=TODAY)
    assert isinstance(await w.route_query(Context(w), RouteRequested(query="x y z")), SemanticSearchRequested)


# ---------- single structured steps ----------
def test_check_structured_query(cfg, repo):
    w = wf(cfg, repo)
    ok, _, _ = w.check_structured_query(StructuredQuery(item_types=["decisions"], date_from="2026-11-08"))
    assert ok == []
    errors, _, _ = w.check_structured_query(StructuredQuery(item_types=["bugs"], tool="cursor", limit=500,
                                                            date_from="2026-12-01", date_to="2026-11-01"))
    text = " ".join(errors)
    assert "bugs" in text and "cursor" in text and "limit" in text and "after" in text and "future" in text


async def test_validate_invalid_then_fallback(cfg, repo):
    w = wf(cfg, repo)
    ctx = Context(w)
    bad = StructuredQueryBuilt(query=StructuredQuery(item_types=["bugs"]))
    async with ctx.store.edit_state() as s:
        s.build_attempts = 1
    assert isinstance(await w.validate_structured_query(ctx, bad), StructuredQueryInvalid)
    async with ctx.store.edit_state() as s:
        s.build_attempts = 2
    out = await w.validate_structured_query(ctx, bad)
    assert isinstance(out, FallbackRequested) and out.reason == "invalid_structured_query"


async def test_execute_hits_and_misses(cfg, repo):
    w = wf(cfg, repo)
    out = await w.execute_structured_query(StructuredQueryValid(query=StructuredQuery(item_types=["decisions"])))
    assert isinstance(out, StructuredResultsFound) and len(out.records) == 1
    out = await w.execute_structured_query(StructuredQueryValid(query=StructuredQuery(tags=["nonexistent"])))
    assert isinstance(out, FallbackRequested) and out.reason == "no_structured_results"


async def test_build_passes_today_and_feedback(cfg, repo):
    w = wf(cfg, repo, structured=[StructuredQuery()])
    ctx = Context(w)
    async with ctx.store.edit_state() as s:
        s.original_query = "q"
    await w.build_structured_query(ctx, StructuredQueryInvalid(errors=["limit too big"]))
    call = w._llm.structured_calls[0]
    assert call["today"] == TODAY.isoformat() and "limit too big" in call["feedback"]


# ---------- structured paths end-to-end ----------
async def test_structured_success(cfg, repo):
    w = wf(cfg, repo, structured=[StructuredQuery(item_types=["decisions", "rules"])],
           texts=["החלטה [1], כלל [2]"])
    r = await w.run(query="תן לי את כל ההחלטות והכללים")
    assert r.route == "structured" and r.grounded and len(r.sources) == 2 and r.sources[0].date
    assert w._index.calls == []           # Pinecone never touched


async def test_structured_invalid_twice_falls_back(cfg, repo):
    bad = StructuredQuery(item_types=["bugs"])
    w = wf(cfg, repo, structured=[bad, ValueError("unparseable")],
           results=[[make_node("a", 0.9), make_node("b", 0.8)]], texts=["answer [1]"])
    r = await w.run(query="something odd")
    assert r.route == "fallback" and "שאילתה מובנית" in r.notice and r.grounded
    assert len(w._llm.structured_calls) == 2   # exactly one rebuild


async def test_structured_no_results_falls_back(cfg, repo):
    w = wf(cfg, repo, structured=[StructuredQuery(date_from="2026-11-08")],     # store has nothing that recent
           results=[[make_node("a", 0.9), make_node("b", 0.8)]], texts=["answer [1]"])
    r = await w.run(query="מה סומן כרגיש בשבוע האחרון?")
    assert r.route == "fallback" and "לא נמצאו פריטים" in r.notice and len(w._index.calls) == 1


def test_tag_match_uses_path_and_header_normalized(repo):
    from rag.structured.schema import StructuredQuery
    hits = repo.query(StructuredQuery(tags=["Frontend_Gateway"]))      # only in the file name
    assert [i.item_type for i in hits] == ["changes"]


async def test_rate_limit_is_retried(monkeypatch):
    import rag.llm as llm_mod

    class TooManyRequestsError(Exception):
        pass
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TooManyRequestsError()
        return "ok"

    async def no_sleep(_):
        return None
    monkeypatch.setattr("asyncio.sleep", no_sleep)
    assert await llm_mod.retry_on_rate_limit(flaky) == "ok" and len(calls) == 3
