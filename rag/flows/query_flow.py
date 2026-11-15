"""Query Workflow.

validate_query -> route_query (Router/Selector)
  semantic:   retrieve -> evaluate_retrieval -> (plan_retry -> rephrase_query | expand) -> synthesize -> validate_answer
  structured: build_structured_query -> validate_structured_query -> execute_structured_query
              -> synthesize_structured -> validate_structured_answer
  fallback:   invalid structured query (after a rebuild) or no structured results -> start_fallback -> semantic route

LLM is called only in: route_query (selector), rephrase_query, synthesize, build_structured_query,
synthesize_structured. Routing is by the event type each step returns — no central dispatcher.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import yaml
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, NodeWithScore
from llama_index.core.tools import ToolMetadata
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field

from rag.embedding import get_embed_model
from rag.flows.logging_utils import emit, log_enter
from rag.loader import PROJECT_ROOT
from rag.structured.prompts import BUILD_QUERY_PROMPT, STRUCTURED_ANSWER_PROMPT
from rag.structured.repository import StructuredRepository
from rag.structured.schema import ITEM_MODELS, ITEM_TYPES, BaseItem, StructuredQuery

logger = logging.getLogger("rag.flows.query")

ROUTES_FILE = PROJECT_ROOT / "prompts" / "routes.yaml"
ROUTE_NAMES = ("semantic", "structured")       # index order of the selector choices

NOT_ENOUGH_INFO = "לא נמצא מידע מספיק בקבצים כדי לענות על השאלה."
QUERY_TOO_SHORT = "השאלה ריקה או קצרה מדי. נסו לנסח שאלה מלאה."
NOT_ENOUGH_INFO_EN = "The files don't contain enough information to answer this question."
QUERY_TOO_SHORT_EN = "The question is empty or too short. Please write a full question."
_CANNED_EN = {NOT_ENOUGH_INFO: NOT_ENOUGH_INFO_EN, QUERY_TOO_SHORT: QUERY_TOO_SHORT_EN}


def is_hebrew(text: str) -> bool:
    return bool(re.search(r"[֐-׿]", text))
FALLBACK_NOTICE = {
    "invalid_structured_query": "לא הצלחתי לבנות שאילתה מובנית תקינה, ולכן עברתי לחיפוש סמנטי.",
    "no_structured_results": "לא נמצאו פריטים מובנים שמתאימים לשאלה, ולכן עברתי לחיפוש סמנטי.",
}

SYNTHESIS_PROMPT = """You answer questions using ONLY the numbered context passages below.
Rules:
- Answer in {language}.
- Use only information from the passages. Do not add general knowledge.
- If the passages do not contain the answer, reply with exactly NO_ANSWER and nothing else.
- If they answer only part of the question, answer that part and say what is missing.
- If passages contradict each other, state the contradiction and cite both passages.
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].

Context passages:
{context}

Question: {question}
Answer:"""

REPHRASE_PROMPT = """Rewrite the user's question as a short English search query for a semantic search over
English technical markdown docs (project instructions, skills, integration summaries).
Use different wording and likely technical terms. Do not repeat any of these already-tried queries:
{tried}

Question: {question}
Return only the query, nothing else."""


# ---------- Events: entry + routing ----------
class QueryStart(StartEvent):
    query: str = ""


class RouteRequested(Event):
    query: str                   # validated query, to be routed


# ---------- Events: semantic route (stage B) ----------
class SearchRequested(Event):
    query_text: str              # what to embed and search for this attempt
    top_k: int


class SemanticSearchRequested(Event):
    query_text: str              # first semantic search (from the router or from a fallback)
    top_k: int


class RetrievalDone(Event):
    nodes: list[NodeWithScore]   # results of *this* search


class ContextSufficient(Event):
    nodes: list[NodeWithScore]


class ContextInsufficient(Event):
    reason: str                  # no_results / too_few / low_score / rephrase_failed


class RephraseRequested(Event):
    pass                         # rephrase_query reads question + tried queries from state


class AnswerDrafted(Event):
    answer: str
    nodes: list[NodeWithScore]   # passages numbered [1..n] in the prompt


# ---------- Events: structured route (stage C) ----------
class StructuredQueryRequested(Event):
    question: str


class StructuredQueryInvalid(Event):
    errors: list[str]            # fed back to the LLM for the single rebuild


class StructuredQueryBuilt(Event):
    query: StructuredQuery | None   # None if the LLM output couldn't be parsed
    error: str = ""


class StructuredQueryValid(Event):
    query: StructuredQuery
    date_from: date | None = None
    date_to: date | None = None


class StructuredResultsFound(Event):
    records: list[BaseItem]


class FallbackRequested(Event):
    reason: str                  # invalid_structured_query / no_structured_results


class StructuredAnswerDrafted(Event):
    answer: str
    records: list[BaseItem]        # items numbered [1..n] in the prompt


# ---------- Result ----------
class Source(BaseModel):
    number: int                  # the [n] the answer cites
    tool: str
    relative_path: str
    header: str
    date: str = ""


class QueryResult(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    grounded: bool = False
    stop_reason: str
    route: str = ""              # semantic / structured / fallback
    notice: str = ""             # e.g. why we fell back


# ---------- State ----------
class QueryState(BaseModel):
    original_query: str = ""
    route: str = ""                                   # chosen route; becomes "fallback" on fallback
    route_reason: str = ""                            # selector's explanation (logs/debug)
    # semantic route
    attempt: int = 0
    tried_queries: list[str] = Field(default_factory=list)
    strategies_used: list[str] = Field(default_factory=list)
    collected_nodes: dict[str, NodeWithScore] = Field(default_factory=dict)
    # structured route
    structured_query: StructuredQuery | None = None   # last built query (shown in logs/UI debugging)
    build_attempts: int = 0                           # enforces the single rebuild
    fallback_reason: str = ""
    stop_reason: str = ""


def answer_language(question: str) -> str:
    """Deterministic language instruction: the LLM sometimes answers Hebrew questions in English."""
    if is_hebrew(question):
        return "Hebrew (the question is in Hebrew)"
    return "the same language as the question"


def is_no_answer(answer: str) -> bool:
    """The synthesis prompts tell the LLM to reply NO_ANSWER when the sources don't answer the question.
    Such a reply must not be shown as a grounded answer, even if it cites passages."""
    return "NO_ANSWER" in answer


def _cited_numbers(answer: str, n: int) -> list[int]:
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)} & set(range(1, n + 1)))


def load_route_choices(path: Path = ROUTES_FILE) -> list[ToolMetadata]:
    routes = yaml.safe_load(path.read_text(encoding="utf-8"))
    choices = []
    for name in ROUTE_NAMES:
        r = routes[name]
        examples = "\n".join(f"  - {q}" for q in r.get("examples", []))
        choices.append(ToolMetadata(name=name, description=f"{r['description'].strip()}\nExample questions:\n{examples}"))
    return choices


def schema_description() -> str:
    lines = []
    for t in ITEM_TYPES:
        own = [f for f in ITEM_MODELS[t].model_fields if f not in BaseItem.model_fields]
        lines.append(f"- {t}: {', '.join(own)}")
    lines.append("Common fields: summary, tags, area, tool, observed_at (date), status (active/superseded)")
    return "\n".join(lines)


async def select_route(selector, choices: list[ToolMetadata], question: str) -> tuple[str, str]:
    """Run the LlamaIndex selector; returns (route_name, reason). Used by the workflow and the eval."""
    from rag.llm import retry_on_rate_limit

    result = await retry_on_rate_limit(lambda: selector.aselect(choices, question))
    sel = result.selections[0]
    return ROUTE_NAMES[sel.index], sel.reason


class QueryWorkflow(Workflow):
    def __init__(self, config: dict, index=None, llm=None, selector=None,
                 repository: StructuredRepository | None = None, today: date | None = None, **kwargs):
        """All dependencies can be injected (tests); otherwise they are built lazily from config."""
        super().__init__(**kwargs)
        self.cfg = config
        self.qcfg = config["query"]
        self.scfg = config["structured"]
        self._index, self._llm, self._selector, self._repo = index, llm, selector, repository
        self._today = today

    # --- lazy real dependencies ---
    @property
    def today(self) -> date:
        return self._today or date.today()

    def _get_index(self):
        if self._index is None:
            from rag.vector_store import get_vector_store

            vs = get_vector_store(self.cfg["pinecone"], self.cfg["embedding"]["dimension"])
            self._index = VectorStoreIndex.from_vector_store(
                vs, embed_model=get_embed_model(self.cfg["embedding"], for_queries=True))
        return self._index

    def _get_llm(self):
        if self._llm is None:
            from rag.llm import get_llm

            self._llm = get_llm(self.cfg)
        return self._llm

    def _get_selector(self):
        if self._selector is None:
            from llama_index.core.selectors import LLMSingleSelector

            self._selector = LLMSingleSelector.from_defaults(llm=self._get_llm())
        return self._selector

    def _get_repo(self) -> StructuredRepository:
        if self._repo is None:
            self._repo = StructuredRepository(PROJECT_ROOT / self.scfg["store_path"])
        return self._repo

    async def _chat(self, prompt: str) -> str:
        """Single-turn LLM call via the chat API. (Cohere's acomplete() uses the legacy /generate
        endpoint, which Command A models don't support.)"""
        from llama_index.core.llms import ChatMessage

        from rag.llm import retry_on_rate_limit

        resp = await retry_on_rate_limit(lambda: self._get_llm().achat([ChatMessage(role="user", content=prompt)]))
        return resp.message.content or ""

    async def _stop(self, ctx: Context[QueryState], step_name: str, result: QueryResult, why: str) -> StopEvent:
        async with ctx.store.edit_state() as s:
            s.stop_reason = result.stop_reason
            route, fb, question = s.route, s.fallback_reason, s.original_query
        if not is_hebrew(question):          # fixed messages follow the question's language
            result.answer = _CANNED_EN.get(result.answer, result.answer)
        result.route = result.route or route
        if fb and not result.notice:
            result.notice = FALLBACK_NOTICE.get(fb, fb)
        return emit(logger, step_name, StopEvent(result=result), why)

    # ================= entry + routing =================
    @step
    async def validate_query(self, ctx: Context[QueryState], ev: QueryStart) -> RouteRequested | StopEvent:
        """Receives: QueryStart(query).
        Returns: RouteRequested(query) for a usable question.
        May emit: StopEvent(stop_reason='invalid_query') for empty/too-short input — no LLM, no Pinecone."""
        log_enter(logger, "validate_query", ev)
        query = (ev.query or "").strip()
        async with ctx.store.edit_state() as s:
            s.original_query = query
        if len(query) < self.qcfg["min_query_chars"]:
            return await self._stop(ctx, "validate_query",
                                    QueryResult(answer=QUERY_TOO_SHORT, stop_reason="invalid_query"),
                                    f"query length {len(query)} < {self.qcfg['min_query_chars']}")
        return emit(logger, "validate_query", RouteRequested(query=query), "query valid")

    @step
    async def route_query(self, ctx: Context[QueryState], ev: RouteRequested) -> SemanticSearchRequested | StructuredQueryRequested:
        """Receives: RouteRequested.  [LLM via LlamaIndex LLMSingleSelector]
        Chooses between the route descriptions in prompts/routes.yaml.
        Returns: SemanticSearchRequested (existing stage-B route) or StructuredQueryRequested.
        If the selector fails, defaults to semantic (the route that always works)."""
        log_enter(logger, "route_query", ev)
        try:
            route, reason = await select_route(self._get_selector(), load_route_choices(), ev.query)
        except Exception as e:
            route, reason = "semantic", f"selector failed ({type(e).__name__}), default to semantic"
            logger.warning("[route_query] %s", reason)
        async with ctx.store.edit_state() as s:
            s.route, s.route_reason = route, reason
        if route == "structured":
            return emit(logger, "route_query", StructuredQueryRequested(question=ev.query), f"structured: {reason}")
        return emit(logger, "route_query", SemanticSearchRequested(query_text=ev.query, top_k=self.qcfg["top_k"]),
                    f"semantic: {reason}")

    # ================= semantic route =================
    @step
    async def retrieve(self, ctx: Context[QueryState], ev: SearchRequested | SemanticSearchRequested) -> RetrievalDone:
        """Receives: SemanticSearchRequested (first search) or SearchRequested (retry).
        Embeds the query (Cohere, search_query) and searches Pinecone; merges results into state.
        Returns: RetrievalDone(nodes). No LLM."""
        log_enter(logger, "retrieve", ev)
        retriever = self._get_index().as_retriever(similarity_top_k=ev.top_k)
        nodes = await retriever.aretrieve(ev.query_text)
        async with ctx.store.edit_state() as s:
            s.attempt += 1
            s.tried_queries.append(ev.query_text)
            for n in nodes:
                prev = s.collected_nodes.get(n.node.node_id)
                if prev is None or (n.score or 0) > (prev.score or 0):
                    s.collected_nodes[n.node.node_id] = n
            attempt = s.attempt
        return emit(logger, "retrieve", RetrievalDone(nodes=nodes),
                    f"attempt {attempt}: {len(nodes)} results for '{ev.query_text}' (top_k={ev.top_k})")

    @step
    async def evaluate_retrieval(self, ctx: Context[QueryState], ev: RetrievalDone) -> ContextSufficient | ContextInsufficient:
        """Receives: RetrievalDone.
        Judges all results collected so far: at least `min_results` with score >= `min_score`.
        Returns: ContextSufficient(nodes) or ContextInsufficient(reason). No LLM."""
        log_enter(logger, "evaluate_retrieval", ev)
        collected = list((await ctx.store.get_state()).collected_nodes.values())
        good = sorted((n for n in collected if (n.score or 0) >= self.qcfg["min_score"]),
                      key=lambda n: n.score or 0, reverse=True)
        if len(good) >= self.qcfg["min_results"]:
            good = good[: self.qcfg["expanded_top_k"]]
            return emit(logger, "evaluate_retrieval", ContextSufficient(nodes=good),
                        f"{len(good)} results >= {self.qcfg['min_score']}")
        if not collected:
            reason = "no_results"
        elif len(collected) < self.qcfg["min_results"]:
            reason = "too_few"
        else:
            reason = "low_score"
        best = max((n.score or 0 for n in collected), default=0)
        return emit(logger, "evaluate_retrieval", ContextInsufficient(reason=reason),
                    f"{reason}: {len(good)} good of {len(collected)}, best score {best:.3f}")

    @step
    async def plan_retry(self, ctx: Context[QueryState], ev: ContextInsufficient) -> RephraseRequested | SearchRequested | StopEvent:
        """Receives: ContextInsufficient(reason).
        Picks the next unused strategy from config `retry_strategies`, if attempts remain.
        Returns: RephraseRequested ('rephrase') or SearchRequested with expanded_top_k ('expand').
        May emit: StopEvent(stop_reason='attempts_exhausted') with the honest 'not enough info' answer. No LLM."""
        log_enter(logger, "plan_retry", ev)
        state = await ctx.store.get_state()
        remaining = [s for s in self.qcfg["retry_strategies"] if s not in state.strategies_used]
        if state.attempt >= self.qcfg["max_attempts"] or not remaining:
            return await self._stop(ctx, "plan_retry",
                                    QueryResult(answer=NOT_ENOUGH_INFO, stop_reason="attempts_exhausted"),
                                    f"attempt {state.attempt}/{self.qcfg['max_attempts']}, strategies left: {remaining}")
        strategy = remaining[0]
        async with ctx.store.edit_state() as s:
            s.strategies_used.append(strategy)
        if strategy == "rephrase":
            return emit(logger, "plan_retry", RephraseRequested(), f"{ev.reason} -> try rephrasing")
        if strategy == "expand":
            return emit(logger, "plan_retry",
                        SearchRequested(query_text=state.original_query, top_k=self.qcfg["expanded_top_k"]),
                        f"{ev.reason} -> expand top_k to {self.qcfg['expanded_top_k']}")
        raise ValueError(f"Unknown retry strategy in config: {strategy}")

    @step
    async def rephrase_query(self, ctx: Context[QueryState], ev: RephraseRequested) -> SearchRequested | ContextInsufficient:
        """Receives: RephraseRequested.  [LLM]
        Returns: SearchRequested(new English query).
        May emit: ContextInsufficient('rephrase_failed') if the LLM returns empty or an already-tried query."""
        log_enter(logger, "rephrase_query", ev)
        state = await ctx.store.get_state()
        prompt = REPHRASE_PROMPT.format(question=state.original_query,
                                        tried="\n".join(f"- {q}" for q in state.tried_queries))
        new_query = (await self._chat(prompt)).strip().strip('"')
        if not new_query or new_query.lower() in {q.lower() for q in state.tried_queries}:
            return emit(logger, "rephrase_query", ContextInsufficient(reason="rephrase_failed"),
                        f"LLM returned unusable query '{new_query}'")
        return emit(logger, "rephrase_query", SearchRequested(query_text=new_query, top_k=self.qcfg["top_k"]),
                    f"rephrased to '{new_query}'")

    @step
    async def synthesize(self, ctx: Context[QueryState], ev: ContextSufficient) -> AnswerDrafted:
        """Receives: ContextSufficient(nodes).  [LLM]
        Answers only from the numbered passages, in the question's language, citing [n].
        Returns: AnswerDrafted(answer, nodes)."""
        log_enter(logger, "synthesize", ev)
        state = await ctx.store.get_state()
        context = "\n\n".join(
            f"[{i}] ({n.node.metadata.get('relative_path')} — {n.node.metadata.get('header')})\n"
            f"{n.node.get_content(metadata_mode=MetadataMode.NONE)}"
            for i, n in enumerate(ev.nodes, 1))
        answer = (await self._chat(
            SYNTHESIS_PROMPT.format(context=context, question=state.original_query,
                                    language=answer_language(state.original_query)))).strip()
        return emit(logger, "synthesize", AnswerDrafted(answer=answer, nodes=ev.nodes),
                    f"{len(answer)} chars from {len(ev.nodes)} passages")

    @step
    async def validate_answer(self, ctx: Context[QueryState], ev: AnswerDrafted) -> StopEvent:
        """Receives: AnswerDrafted.
        Accepts the answer only if non-empty and citing at least one real passage [n].
        Returns: StopEvent(grounded answer + sources) or StopEvent('not_grounded', honest answer)."""
        log_enter(logger, "validate_answer", ev)
        cited = _cited_numbers(ev.answer, len(ev.nodes))
        if not ev.answer or not cited or is_no_answer(ev.answer):
            return await self._stop(ctx, "validate_answer",
                                    QueryResult(answer=NOT_ENOUGH_INFO, stop_reason="not_grounded"),
                                    "empty answer" if not ev.answer else "LLM said the passages don't answer" if is_no_answer(ev.answer) else "answer cites no passage")
        sources = []
        for i in cited:
            md = ev.nodes[i - 1].node.metadata
            sources.append(Source(number=i, tool=md.get("tool", ""), relative_path=md.get("relative_path", ""),
                                  header=md.get("header", ""), date=md.get("last_modified_date", "")))
        return await self._stop(ctx, "validate_answer",
                                QueryResult(answer=ev.answer, sources=sources, grounded=True, stop_reason="answered"),
                                f"grounded in {len(cited)} passages")

    # ================= structured route =================
    @step
    async def build_structured_query(self, ctx: Context[QueryState], ev: StructuredQueryRequested | StructuredQueryInvalid) -> StructuredQueryBuilt:
        """Receives: StructuredQueryRequested (first build) or StructuredQueryInvalid (the single rebuild,
        with the validation errors as feedback).  [LLM]
        The LLM gets the schema, today's date and the question, and returns a StructuredQuery (Pydantic,
        no free code/SQL). Returns: StructuredQueryBuilt(query) — query=None + error if unparseable."""
        log_enter(logger, "build_structured_query", ev)
        state = await ctx.store.get_state()
        feedback = ""
        if isinstance(ev, StructuredQueryInvalid):
            feedback = "Your previous query was invalid: " + "; ".join(ev.errors) + ". Fix these problems.\n"
        async with ctx.store.edit_state() as s:
            s.build_attempts += 1
        try:
            from rag.llm import retry_on_rate_limit

            q = await retry_on_rate_limit(lambda: self._get_llm().astructured_predict(
                StructuredQuery, BUILD_QUERY_PROMPT,
                today=self.today.isoformat(), schema=schema_description(), tools=", ".join(self.cfg["tools"]),
                max_limit=self.scfg["max_limit"], feedback=feedback, question=state.original_query))
        except Exception as e:
            return emit(logger, "build_structured_query", StructuredQueryBuilt(query=None, error=f"{type(e).__name__}: {e}"[:300]),
                        "LLM output could not be parsed")
        async with ctx.store.edit_state() as s:
            s.structured_query = q
        return emit(logger, "build_structured_query", StructuredQueryBuilt(query=q), f"built {q.model_dump(exclude_defaults=True)}")

    def check_structured_query(self, q: StructuredQuery) -> tuple[list[str], date | None, date | None]:
        """Pure validation (no LLM, no IO). Returns (errors, date_from, date_to)."""
        errors: list[str] = []
        bad_types = [t for t in q.item_types if t not in ITEM_TYPES]
        if bad_types:
            errors.append(f"unknown item_types {bad_types}; allowed {list(ITEM_TYPES)}")
        if q.tool and q.tool not in self.cfg["tools"]:
            errors.append(f"unknown tool '{q.tool}'; allowed {self.cfg['tools']}")
        if q.sort not in ("newest", "oldest"):
            errors.append(f"sort must be 'newest' or 'oldest', got '{q.sort}'")
        if not 1 <= q.limit <= self.scfg["max_limit"]:
            errors.append(f"limit must be 1..{self.scfg['max_limit']}, got {q.limit}")
        parsed: dict[str, date | None] = {}
        for name in ("date_from", "date_to"):
            raw = getattr(q, name)
            try:
                parsed[name] = date.fromisoformat(raw) if raw else None
            except ValueError:
                errors.append(f"{name} '{raw}' is not an ISO date YYYY-MM-DD")
                parsed[name] = None
        d_from, d_to = parsed["date_from"], parsed["date_to"]
        earliest = date.fromisoformat(self.scfg["earliest_date"])
        if d_from and d_to and d_from > d_to:
            errors.append(f"date_from {d_from} is after date_to {d_to}")
        if d_from and d_from > self.today:
            errors.append(f"date_from {d_from} is in the future (today is {self.today})")
        for d in (d_from, d_to):
            if d and d < earliest:
                errors.append(f"date {d} is before {earliest}")
        return errors, d_from, d_to

    @step
    async def validate_structured_query(self, ctx: Context[QueryState], ev: StructuredQueryBuilt) -> StructuredQueryValid | StructuredQueryInvalid | FallbackRequested:
        """Receives: StructuredQueryBuilt.
        Checks: item_types/tool exist in the schema/config, dates parse and make sense, limit in range.
        Returns: StructuredQueryValid(query, parsed dates).
        May emit: StructuredQueryInvalid(errors) -> one rebuild; after max_build_attempts ->
        FallbackRequested('invalid_structured_query'). No LLM."""
        log_enter(logger, "validate_structured_query", ev)
        if ev.query is None:
            errors, d_from, d_to = [f"unparseable output: {ev.error}"], None, None
        else:
            errors, d_from, d_to = self.check_structured_query(ev.query)
        if not errors:
            return emit(logger, "validate_structured_query",
                        StructuredQueryValid(query=ev.query, date_from=d_from, date_to=d_to), "query valid")
        attempts = (await ctx.store.get_state()).build_attempts
        if attempts < self.scfg["max_build_attempts"]:
            return emit(logger, "validate_structured_query", StructuredQueryInvalid(errors=errors),
                        f"invalid ({'; '.join(errors)}) -> rebuild {attempts + 1}/{self.scfg['max_build_attempts']}")
        return emit(logger, "validate_structured_query", FallbackRequested(reason="invalid_structured_query"),
                    f"still invalid after {attempts} builds: {'; '.join(errors)}")

    @step
    async def execute_structured_query(self, ev: StructuredQueryValid) -> StructuredResultsFound | FallbackRequested:
        """Receives: StructuredQueryValid.
        Runs the query on the repository (hard filters, full coverage up to limit). No LLM.
        Returns: StructuredResultsFound(items).  May emit: FallbackRequested('no_structured_results')."""
        log_enter(logger, "execute_structured_query", ev)
        items = self._get_repo().query(ev.query, ev.date_from, ev.date_to)
        if not items:
            return emit(logger, "execute_structured_query", FallbackRequested(reason="no_structured_results"),
                        "0 items matched")
        return emit(logger, "execute_structured_query", StructuredResultsFound(records=items), f"{len(items)} items")

    @step
    async def start_fallback(self, ctx: Context[QueryState], ev: FallbackRequested) -> SemanticSearchRequested:
        """Receives: FallbackRequested(reason).
        Records route='fallback' and the reason in state (shown to the user), then hands the original
        question to the semantic route. Returns: SemanticSearchRequested. No LLM."""
        log_enter(logger, "start_fallback", ev)
        async with ctx.store.edit_state() as s:
            s.route, s.fallback_reason = "fallback", ev.reason
            q = s.original_query
        return emit(logger, "start_fallback", SemanticSearchRequested(query_text=q, top_k=self.qcfg["top_k"]),
                    f"{ev.reason} -> semantic search")

    @step
    async def synthesize_structured(self, ctx: Context[QueryState], ev: StructuredResultsFound) -> StructuredAnswerDrafted:
        """Receives: StructuredResultsFound(items).  [LLM]
        Answers only from the numbered items, in the question's language, citing [n].
        Returns: StructuredAnswerDrafted(answer, items)."""
        log_enter(logger, "synthesize_structured", ev)
        state = await ctx.store.get_state()
        lines = []
        for i, it in enumerate(ev.records, 1):
            lines.append(f"[{i}] ({it.item_type}, {it.status}, {it.observed_at}, {it.source.tool}, "
                         f"{it.source.relative_path} — {it.source.header})\n{it.summary}\n{it.key_text()}")
        answer = (await self._chat(
            STRUCTURED_ANSWER_PROMPT.format(items="\n\n".join(lines), question=state.original_query,
                                            language=answer_language(state.original_query)))).strip()
        return emit(logger, "synthesize_structured", StructuredAnswerDrafted(answer=answer, records=ev.records),
                    f"{len(answer)} chars from {len(ev.records)} items")

    @step
    async def validate_structured_answer(self, ctx: Context[QueryState], ev: StructuredAnswerDrafted) -> StopEvent:
        """Receives: StructuredAnswerDrafted.
        Same grounding rule as the semantic route: non-empty and cites at least one item.
        Returns: StopEvent(answer + sources with tool, path, header, date) or StopEvent('not_grounded')."""
        log_enter(logger, "validate_structured_answer", ev)
        cited = _cited_numbers(ev.answer, len(ev.records))
        if not ev.answer or not cited or is_no_answer(ev.answer):
            return await self._stop(ctx, "validate_structured_answer",
                                    QueryResult(answer=NOT_ENOUGH_INFO, stop_reason="not_grounded"),
                                    "empty answer" if not ev.answer else "LLM said the items don't answer" if is_no_answer(ev.answer) else "answer cites no item")
        sources = [Source(number=i, tool=ev.records[i - 1].source.tool,
                          relative_path=ev.records[i - 1].source.relative_path,
                          header=ev.records[i - 1].source.header,
                          date=ev.records[i - 1].observed_at.isoformat()) for i in cited]
        return await self._stop(ctx, "validate_structured_answer",
                                QueryResult(answer=ev.answer, sources=sources, grounded=True, stop_reason="answered"),
                                f"grounded in {len(cited)} items")


ROUTE_LABELS = {"semantic": "חיפוש סמנטי", "structured": "שאילתה מובנית", "fallback": "מובנה ← fallback לסמנטי"}


def format_result(result: QueryResult) -> str:
    """Markdown for the UI: route, optional notice, answer, sources."""
    lines = []
    if result.route:
        lines.append(f"*מסלול: {ROUTE_LABELS.get(result.route, result.route)}*")
    if result.notice:
        lines.append(f"> {result.notice}")
    lines += ["", result.answer] if lines else [result.answer]
    if result.sources:
        lines += ["", "**מקורות:**"]
        lines += [f"- [{s.number}] `{s.tool}` · `{s.relative_path}` · {s.header}" + (f" · {s.date}" if s.date else "")
                  for s in result.sources]
    return "\n".join(lines)
