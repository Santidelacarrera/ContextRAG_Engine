"""Regression: vectors of another size or model are refused with a clear error, never stored, never silently mixed."""
import pytest

from contextrag import factory
from contextrag.core.chunking import chunk_document
from contextrag.core.config import Settings
from contextrag.core.embeddings import HashingEmbedder
from contextrag.retrieval.vector_store import DimensionMismatch, InMemoryVectorStore

TEXT = "El plazo es de cuatro años contados desde la notificación."


def chunks(doc_id, text=TEXT):
    return chunk_document(text, doc_id=doc_id, tenant="t", source=doc_id, kind="text", acl=(), page_starts=None,
                          settings=Settings())


def test_an_index_refuses_vectors_of_another_size_with_an_explanation():
    r = factory.offline()
    r.ingestor.ingest_text(TEXT, tenant="t", doc_id="a.md", source="a.md")
    r.ingestor.embedder = HashingEmbedder(dim=64)
    with pytest.raises(DimensionMismatch, match="re-ingest the documents or erase the tenant"):
        r.ingestor.ingest_text("Otro documento distinto sobre plazos.", tenant="t", doc_id="b.md", source="b.md")


def test_the_refused_write_leaves_the_index_exactly_as_it_was():
    r = factory.offline()
    r.ingestor.ingest_text(TEXT, tenant="t", doc_id="a.md", source="a.md")
    before = [d["doc_id"] for d in r.index.list_documents("t")]
    r.ingestor.embedder = HashingEmbedder(dim=64)
    with pytest.raises(DimensionMismatch):
        r.ingestor.ingest_text("Otro documento distinto sobre plazos.", tenant="t", doc_id="b.md", source="b.md")
    assert [d["doc_id"] for d in r.index.list_documents("t")] == before
    assert r.ask("¿Cuál es el plazo?", tenant="t").status.value in ("verified", "fallback")  # still answers


def test_one_write_with_mixed_sizes_is_refused():
    store = InMemoryVectorStore()
    p, c = chunks("long.md", " ".join(f"palabra{i}" for i in range(600)))  # several children, so the batch is real
    assert len(c) >= 2
    vectors = [[0.1] * 8 if i == 0 else [0.1] * 9 for i in range(len(c))]
    with pytest.raises(DimensionMismatch, match="different sizes in one write"):
        store.add(p, c, vectors)
    assert store._children == {}  # nothing from the refused batch was stored


def test_the_qdrant_store_refuses_a_wrong_size_vector_and_writes_nothing():
    qdrant_client = pytest.importorskip("qdrant_client")
    from contextrag.retrieval.vector_store import QdrantVectorStore

    client = qdrant_client.QdrantClient(":memory:")
    store = QdrantVectorStore(512, client=client)
    p, c = chunks("a.md")
    store.add(p, c, HashingEmbedder(512).embed([x.text for x in c]))
    before = client.count(store.col).count
    p2, c2 = chunks("b.md", "Otro documento distinto.")
    with pytest.raises(Exception, match="expected dim: 512"):
        store.add(p2, c2, HashingEmbedder(64).embed([x.text for x in c2]))
    assert client.count(store.col).count == before  # the batch was rejected as a whole
