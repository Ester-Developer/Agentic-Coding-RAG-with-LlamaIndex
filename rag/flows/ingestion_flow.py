"""Ingestion Workflow: load -> validate -> chunk -> validate -> prepare index -> embed -> store.

No step here calls an LLM. Routing is by event type: each step declares the
events it can return, and the framework delivers each event to the step that accepts it.
"""
from __future__ import annotations

import logging
from pathlib import Path

from llama_index.core import Document
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel, Field

from rag.chunking import chunk_documents
from rag.embedding import get_embed_model
from rag.flows.logging_utils import emit, log_enter
from rag.loader import DEFAULT_CONFIG, ConfigError, load_config, load_documents
from rag.vector_store import DimensionMismatchError, get_vector_store

logger = logging.getLogger("rag.flows.ingestion")

REQUIRED_METADATA = ("tool", "relative_path")


# ---------- Events (only what the next step needs) ----------
class IngestStart(StartEvent):
    config_path: str = str(DEFAULT_CONFIG)


class DocumentsLoaded(Event):
    documents: list[Document]


class DocumentsValidated(Event):
    documents: list[Document]


class NodesChunked(Event):
    nodes: list[BaseNode]


class NodesValidated(Event):
    nodes: list[BaseNode]


class IndexReady(Event):
    nodes: list[BaseNode]


class NodesEmbedded(Event):
    nodes: list[BaseNode]


class IngestSummary(BaseModel):
    ok: bool
    documents: int = 0
    nodes: int = 0
    skipped: list[str] = Field(default_factory=list)
    error: str = ""


# ---------- State (needed across the whole run) ----------
class IngestState(BaseModel):
    config: dict = Field(default_factory=dict)          # read by chunk/prepare_index steps
    skipped: list[str] = Field(default_factory=list)    # appended by both validation steps, reported at the end
    documents_count: int = 0                            # for the final summary


class IngestionWorkflow(Workflow):
    def __init__(self, embed_model=None, vector_store=None, **kwargs):
        """embed_model / vector_store can be injected (tests); otherwise built from config."""
        super().__init__(**kwargs)
        self._embed_model = embed_model
        self._vector_store = vector_store

    @staticmethod
    def _fail(summary_error: str, step_name: str) -> StopEvent:
        return emit(logger, step_name, StopEvent(result=IngestSummary(ok=False, error=summary_error)), "fatal error")

    @step
    async def load(self, ctx: Context[IngestState], ev: IngestStart) -> DocumentsLoaded | StopEvent:
        """Receives: IngestStart(config_path).
        Returns: DocumentsLoaded(documents) — one Document per configured file.
        May emit: StopEvent(ok=False) if config is invalid or a source file is missing."""
        log_enter(logger, "load", ev)
        try:
            cfg = load_config(ev.config_path)
            docs = load_documents(ev.config_path)
        except (FileNotFoundError, ConfigError) as e:
            return self._fail(str(e), "load")
        async with ctx.store.edit_state() as s:
            s.config = cfg
        return emit(logger, "load", DocumentsLoaded(documents=docs), f"{len(docs)} documents loaded")

    @step
    async def validate_documents(self, ctx: Context[IngestState], ev: DocumentsLoaded) -> DocumentsValidated | StopEvent:
        """Receives: DocumentsLoaded.
        Returns: DocumentsValidated(documents) without empty files / docs missing tool or relative_path.
        May emit: StopEvent(ok=False) if no valid document is left. Skips are logged as warnings."""
        log_enter(logger, "validate_documents", ev)
        valid, skipped = [], []
        for doc in ev.documents:
            name = doc.metadata.get("relative_path") or doc.metadata.get("file_name", "<unknown>")
            missing = [k for k in REQUIRED_METADATA if not doc.metadata.get(k)]
            if missing:
                reason = f"{name}: missing metadata {missing}"
            elif not doc.text.strip():
                reason = f"{name}: empty file"
            else:
                valid.append(doc)
                continue
            logger.warning("[validate_documents] skipping %s", reason)
            skipped.append(reason)
        async with ctx.store.edit_state() as s:
            s.skipped.extend(skipped)
            s.documents_count = len(valid)
        if not valid:
            return self._fail("No valid documents to ingest", "validate_documents")
        return emit(logger, "validate_documents", DocumentsValidated(documents=valid),
                    f"{len(valid)} valid, {len(skipped)} skipped")

    @step
    async def chunk(self, ctx: Context[IngestState], ev: DocumentsValidated) -> NodesChunked:
        """Receives: DocumentsValidated.
        Returns: NodesChunked(nodes) — Markdown header split, then SentenceSplitter for long sections.
        May emit: only NodesChunked."""
        log_enter(logger, "chunk", ev)
        cfg = (await ctx.store.get_state()).config
        nodes = chunk_documents(ev.documents, **cfg["chunking"])
        return emit(logger, "chunk", NodesChunked(nodes=nodes), f"{len(nodes)} nodes")

    @step
    async def validate_nodes(self, ctx: Context[IngestState], ev: NodesChunked) -> NodesValidated | StopEvent:
        """Receives: NodesChunked.
        Returns: NodesValidated(nodes) without nodes whose text is empty/whitespace.
        May emit: StopEvent(ok=False) if no node is left. Skips are logged as warnings."""
        log_enter(logger, "validate_nodes", ev)
        valid, skipped = [], []
        for node in ev.nodes:
            if node.get_content(metadata_mode=MetadataMode.NONE).strip():
                valid.append(node)
            else:
                reason = f"node {node.id_}: empty after chunking"
                logger.warning("[validate_nodes] skipping %s", reason)
                skipped.append(reason)
        async with ctx.store.edit_state() as s:
            s.skipped.extend(skipped)
        if not valid:
            return self._fail("No non-empty nodes after chunking", "validate_nodes")
        return emit(logger, "validate_nodes", NodesValidated(nodes=valid), f"{len(valid)} valid, {len(skipped)} skipped")

    @step
    async def prepare_index(self, ctx: Context[IngestState], ev: NodesValidated) -> IndexReady | StopEvent:
        """Receives: NodesValidated.
        Checks the embedding model's real dimension against config, then creates the Pinecone
        index if missing or verifies dimension/metric if it exists.
        Returns: IndexReady(nodes).  May emit: StopEvent(ok=False) on dimension/metric mismatch or missing API key."""
        log_enter(logger, "prepare_index", ev)
        cfg = (await ctx.store.get_state()).config
        try:
            if self._embed_model is None:
                self._embed_model = get_embed_model(cfg["embedding"])
            dim = len(await self._embed_model.aget_text_embedding("dimension check"))
            if dim != cfg["embedding"]["dimension"]:
                raise DimensionMismatchError(
                    f"config embedding.dimension={cfg['embedding']['dimension']} but model returns {dim}")
            if self._vector_store is None:
                self._vector_store = get_vector_store(cfg["pinecone"], dim)
        except (EnvironmentError, DimensionMismatchError) as e:
            return self._fail(str(e), "prepare_index")
        return emit(logger, "prepare_index", IndexReady(nodes=ev.nodes), f"index ready (dim={dim})")

    @step
    async def embed(self, ev: IndexReady) -> NodesEmbedded:
        """Receives: IndexReady.
        Returns: NodesEmbedded(nodes) with node.embedding set (Cohere, input_type=search_document).
        Only metadata not excluded from embedding is included in the embedded text."""
        log_enter(logger, "embed", ev)
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in ev.nodes]
        vectors = await self._embed_model.aget_text_embedding_batch(texts)
        for node, vec in zip(ev.nodes, vectors):
            node.embedding = vec
        return emit(logger, "embed", NodesEmbedded(nodes=ev.nodes), f"{len(vectors)} vectors")

    @step
    async def store(self, ctx: Context[IngestState], ev: NodesEmbedded) -> StopEvent:
        """Receives: NodesEmbedded.
        Deletes each source file's old vectors (by ref_doc_id), then upserts with stable ids -> no duplicates.
        Returns: StopEvent(result=IngestSummary)."""
        log_enter(logger, "store", ev)
        for doc_id in sorted({n.ref_doc_id for n in ev.nodes}):
            self._vector_store.delete(ref_doc_id=doc_id)
        self._vector_store.add(ev.nodes)
        state = await ctx.store.get_state()
        summary = IngestSummary(ok=True, documents=state.documents_count, nodes=len(ev.nodes), skipped=state.skipped)
        return emit(logger, "store", StopEvent(result=summary), f"{len(ev.nodes)} vectors upserted")


async def run_ingestion(config_path: str | Path = DEFAULT_CONFIG, **deps) -> IngestSummary:
    wf = IngestionWorkflow(timeout=600, **deps)
    return await wf.run(start_event=IngestStart(config_path=str(config_path)))
