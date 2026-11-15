"""Fakes for Cohere (LLM + embeddings), Pinecone (index/vector store) and the router selector.
No test makes a network call."""
from __future__ import annotations

import copy
from datetime import date
from types import SimpleNamespace

import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.schema import NodeWithScore, TextNode

from rag.loader import load_config
from rag.structured.repository import StructuredRepository
from rag.structured.schema import Change, Decision, Rule, SourceRef, WarningItem, make_id

TODAY = date(2026, 11, 15)


def _sources_available() -> bool:
    from rag.loader import resolve_sources
    try:
        resolve_sources(load_config())
        return True
    except Exception:
        return False


# tests that read the real source markdown files (skipped on a clean clone / CI)
requires_sources = pytest.mark.skipif(not _sources_available(), reason="source project not available (set RAG_SOURCE_ROOT)")


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


# ---------- LLM ----------
class FakeLLM:
    """achat() pops from `texts`; astructured_predict() pops from `structured`
    (an Exception instance in the queue is raised)."""

    def __init__(self, texts=None, structured=None):
        self.texts = list(texts or [])
        self.structured = list(structured or [])
        self.prompts: list[str] = []
        self.structured_calls: list[dict] = []

    async def achat(self, messages, **kw):
        self.prompts.append(messages[-1].content)
        return ChatResponse(message=ChatMessage(role="assistant", content=self.texts.pop(0)))

    async def astructured_predict(self, output_cls, prompt, **kwargs):
        self.structured_calls.append({"cls": output_cls, **kwargs})
        value = self.structured.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


# ---------- Pinecone retrieval ----------
def make_node(node_id: str, score: float, text: str = "some text", **md) -> NodeWithScore:
    meta = {"tool": "claude_code", "relative_path": "CLAUDE.md", "header": "Commands",
            "last_modified_date": "2026-10-25", **md}
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata=meta), score=score)


class FakeIndex:
    """as_retriever().aretrieve() returns the next list from `results` and records each call."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[tuple[str, int]] = []

    def as_retriever(self, similarity_top_k):
        index = self

        class _R:
            async def aretrieve(self, q):
                index.calls.append((q, similarity_top_k))
                return index.results.pop(0)
        return _R()


# ---------- Router ----------
class FakeSelector:
    def __init__(self, index: int, reason: str = "test"):
        self.index, self.reason = index, reason
        self.questions: list[str] = []

    async def aselect(self, choices, query):
        self.questions.append(query)
        return SimpleNamespace(selections=[SimpleNamespace(index=self.index, reason=self.reason)])


SEMANTIC, STRUCTURED = 0, 1


# ---------- Structured store ----------
def _src(path="CLAUDE.md", tool="claude_code", header="Architecture", lines=(1, 5)):
    return SourceRef(tool=tool, relative_path=path, header=header, line_range=lines)


def sample_items():
    items = [
        Decision(topic="kafka_topic_layout", summary="All sources share one Kafka topic.", tags=["kafka"],
                 area="bridge", decision="Single topic bridge.structured.tasks", source=_src(),
                 observed_at=date(2026, 10, 25), date_source="file_mtime"),
        Rule(topic="proto_files", summary="Do not copy proto files into teams.", tags=["proto"], area="relay",
             rule="Don't copy shared/proto files", strength="must", source=_src(),
             observed_at=date(2026, 10, 25), date_source="file_mtime"),
        WarningItem(topic="helm_shared_release", summary="A broken value blocks sync for every team.",
                    tags=["helm", "argocd"], area="relay/helm", risk="One bad value blocks all teams",
                    kind="limitation", source=_src(), observed_at=date(2026, 10, 25), date_source="file_mtime"),
        Change(topic="auth_login_endpoint", summary="Login endpoint moved.", tags=["auth", "endpoint"],
               area="lens/frontend", subject="login endpoint", from_value="/api/auth/login", to_value="/auth/login",
               source=_src("FRONTEND_GATEWAY_CONNECTION_SUMMARY.md", header="2. Auth API Types"),
               observed_at=date(2026, 10, 20), date_source="content"),
    ]
    for i in items:
        i.id = make_id(i)
    return items


@pytest.fixture
def repo(tmp_path):
    r = StructuredRepository(tmp_path / "store.json")
    by_file: dict[str, list] = {}
    for item in sample_items():
        by_file.setdefault(item.source.relative_path, []).append(item)
    for rel, items in by_file.items():
        r.replace_file_items(rel, {"tool": items[0].source.tool, "root_path": "X", "last_modified": "2026-10-25",
                                   "hash": "h"}, items)
    r.recompute_status()
    return r
