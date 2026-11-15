"""Ingestion Workflow steps with fake embedding model and vector store."""
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.workflow import Context

from rag.flows.ingestion_flow import (DocumentsLoaded, IndexReady, IngestionWorkflow, NodesChunked,
                                      NodesEmbedded, NodesValidated)


class FakeEmbed:
    async def aget_text_embedding_batch(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeVectorStore:
    def __init__(self):
        self.deleted, self.added = [], []

    def delete(self, ref_doc_id):
        self.deleted.append(ref_doc_id)

    def add(self, nodes):
        self.added.extend(nodes)


def doc(text, **md):
    return Document(text=text, metadata=md)


async def test_validate_documents_skips_bad():
    w = IngestionWorkflow()
    ctx = Context(w)
    docs = [doc("# ok\ntext", tool="kiro", relative_path="a.md"),
            doc("   ", tool="kiro", relative_path="empty.md"),
            doc("# no tool", relative_path="b.md")]
    out = await w.validate_documents(ctx, DocumentsLoaded(documents=docs))
    assert [d.metadata["relative_path"] for d in out.documents] == ["a.md"]
    assert len((await ctx.store.get_state()).skipped) == 2


async def test_validate_documents_all_bad_stops():
    w = IngestionWorkflow()
    out = await w.validate_documents(Context(w), DocumentsLoaded(documents=[doc("", tool="kiro", relative_path="e.md")]))
    assert out.result.ok is False


async def test_validate_nodes_skips_empty():
    w = IngestionWorkflow()
    nodes = [TextNode(id_="a", text="hello"), TextNode(id_="b", text="  \n")]
    out = await w.validate_nodes(Context(w), NodesChunked(nodes=nodes))
    assert isinstance(out, NodesValidated) and [n.id_ for n in out.nodes] == ["a"]


async def test_embed_and_store_replace_per_file():
    vs = FakeVectorStore()
    w = IngestionWorkflow(embed_model=FakeEmbed(), vector_store=vs)
    node = TextNode(id_="a.md::0000", text="hello", metadata={"relative_path": "a.md"})
    node.relationships = {}
    embedded = await w.embed(IndexReady(nodes=[node]))
    assert isinstance(embedded, NodesEmbedded) and embedded.nodes[0].embedding == [0.1, 0.2]
    out = await w.store(Context(w), embedded)
    assert out.result.ok and vs.added[0].id_ == "a.md::0000"
