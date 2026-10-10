import math

import pytest

from contextrag.core.cache import CachedEmbedder, InMemoryCache
from contextrag.core.chunking import chunk_document
from contextrag.core.config import Settings
from contextrag.core.embeddings import HashingEmbedder
from contextrag.core.models import ChildChunk
from contextrag.retrieval.bm25 import BM25Index
from contextrag.retrieval.hybrid_search import HybridIndex, HybridRetriever, reciprocal_rank_fusion
from contextrag.retrieval.reranker import LexicalReranker
from contextrag.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore

DOCS = {
    "audit.md": "Los registros de auditoría se conservan durante 7 años.\n\nLos registros de acceso se eliminan a los 90 días.",
    "crypto.md": "Todos los datos en reposo se cifran con AES-256.\n\nLas claves de cifrado se rotan cada 12 meses.",
    "cats.md": "Los gatos duermen la mayor parte del día y cazan al anochecer.",
}


def child(i, text, tenant="t", acl=(), doc="d"):
    return ChildChunk(f"c{i}", f"p{i}", doc, tenant, text, 0, len(text), acl)


def build(store=None, tenant="acme", cfg=None):
    emb = HashingEmbedder()
    index = HybridIndex(store or InMemoryVectorStore())
    cfg = cfg or Settings()
    for doc, text in DOCS.items():
        parents, children = chunk_document(text, doc_id=doc, tenant=tenant, source=doc, settings=cfg)
        index.add(parents, children, emb.embed([c.text for c in children]))
    return index, HybridRetriever(index, emb, LexicalReranker(), cfg)


# ---- RRF ---------------------------------------------------------------------------------

def test_rrf_scores_match_formula():
    a, b, c = child(1, "a"), child(2, "b"), child(3, "c")
    fused = dict((x.id, s) for x, s in reciprocal_rank_fusion([[a, b], [b, c]], k=60))
    assert fused["c1"] == pytest.approx(1 / 61)
    assert fused["c2"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c3"] == pytest.approx(1 / 62)
    order = [x.id for x, _ in reciprocal_rank_fusion([[a, b], [b, c]], k=60)]
    assert order == ["c2", "c1", "c3"]  # agreement across rankers wins


def test_rrf_handles_empty_and_single_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    a = child(1, "a")
    assert reciprocal_rank_fusion([[a], []])[0][0] is a


# ---- BM25 --------------------------------------------------------------------------------

def test_bm25_ranks_rare_term_matches_first():
    idx = BM25Index()
    idx.add([child(1, "the cat sat on the mat"), child(2, "quantum chromodynamics lecture"), child(3, "the dog ran")])
    hits = idx.search("quantum lecture", "t", frozenset(), 5)
    assert [c.id for c, _ in hits] == ["c2"]
    assert hits[0][1] > 0


def test_bm25_positive_scores_on_tiny_corpora():
    idx = BM25Index()
    idx.add([child(1, "solo document about apples")])
    assert idx.search("apples", "t", frozenset(), 3)[0][1] > 0  # BM25Okapi would be negative here


def test_bm25_is_tenant_and_acl_scoped_and_supports_delete():
    idx = BM25Index()
    idx.add([child(1, "secret apples", tenant="a", acl=("hr",)), child(2, "public apples", tenant="a", doc="pub"),
             child(3, "other tenant apples", tenant="b")])
    assert [c.id for c, _ in idx.search("apples", "a", frozenset(), 9)] == ["c2"]
    assert {c.id for c, _ in idx.search("apples", "a", frozenset({"hr"}), 9)} == {"c1", "c2"}
    assert [c.id for c, _ in idx.search("apples", "b", frozenset(), 9)] == ["c3"]
    idx.delete_document("a", "pub")
    assert idx.search("apples", "a", frozenset(), 9) == []


def test_bm25_ignores_stopword_only_and_unknown_queries():
    idx = BM25Index()
    idx.add([child(1, "apples and pears")])
    assert idx.search("the of and", "t", frozenset(), 5) == []
    assert idx.search("zzz", "t", frozenset(), 5) == []
    assert BM25Index().search("apples", "nobody", frozenset(), 5) == []


# ---- hybrid retrieval --------------------------------------------------------------------

def test_hybrid_returns_parent_chunks_with_citations_in_score_order():
    _, r = build()
    ctx = r.retrieve("¿cuánto se conservan los registros de auditoría?", "acme")
    assert ctx[0].parent.doc_id == "audit.md" and ctx[0].index == 1
    assert [c.index for c in ctx] == list(range(1, len(ctx) + 1))
    assert all(a.score >= b.score for a, b in zip(ctx, ctx[1:]))
    assert "7 años" in ctx[0].parent.text and "90 días" in ctx[0].parent.text  # whole parent, not just the matching child


def test_dense_branch_contributes_when_bm25_has_no_lexical_overlap():
    class Synonyms:
        dim = 2

        def embed(self, texts):  # "coche" and "automóvil" are synonyms -> same direction
            return [[1.0, 0.0] if ("coche" in t or "automóvil" in t) else [0.0, 1.0] for t in texts]

    class Accept:
        def score(self, query, passages):
            return [1.0] * len(passages)

    index = HybridIndex(InMemoryVectorStore())
    emb = Synonyms()
    for doc, text in {"car.md": "El automóvil necesita revisión anual.", "fruit.md": "La manzana es una fruta."}.items():
        p, c = chunk_document(text, doc_id=doc, tenant="t", source=doc)
        index.add(p, c, emb.embed([x.text for x in c]))
    r = HybridRetriever(index, emb, Accept(), Settings())
    assert index.bm25.search("coche", "t", frozenset(), 5) == []  # sparse finds nothing
    assert r.retrieve("coche", "t", top_k=1)[0].parent.doc_id == "car.md"  # dense does


def test_top_k_and_irrelevant_query():
    _, r = build()
    assert len(r.retrieve("registros cifrado gatos claves", "acme", top_k=2)) == 2
    assert r.retrieve("xyzzy plugh", "acme") == []


def test_tenant_isolation_in_retrieval():
    _, r = build()
    assert r.retrieve("registros de auditoría", "other") == []


def test_acl_filtering_in_retrieval():
    index, r = build()
    cfg = Settings()
    parents, children = chunk_document("El plan de despidos es confidencial.", doc_id="hr.md", tenant="acme",
                                       source="hr.md", acl=("hr",), settings=cfg)
    index.add(parents, children, r.embedder.embed([c.text for c in children]))
    assert r.retrieve("plan de despidos", "acme") == []
    assert r.retrieve("plan de despidos", "acme", frozenset({"hr"}))[0].parent.doc_id == "hr.md"


def test_reindex_replaces_old_content_in_both_indexes():
    index, r = build()
    index.delete_document("acme", "audit.md")
    assert r.retrieve("registros de auditoría 7 años", "acme") == []
    assert index.bm25.search("auditoría", "acme", frozenset(), 5) == []


def test_bm25_rebuilds_from_durable_store_after_restart():
    store = InMemoryVectorStore()
    build(store)
    fresh = HybridIndex(store)  # new process: empty BM25, same durable store
    r = HybridRetriever(fresh, HashingEmbedder(), LexicalReranker(), Settings())
    assert r.retrieve("registros de auditoría", "acme")[0].parent.doc_id == "audit.md"
    assert [c.doc_id for c, _ in fresh.bm25.search("AES-256 cifran", "acme", frozenset(), 3)][0] == "crypto.md"


def test_reranker_reorders_fused_candidates():
    class Reverse:
        def score(self, query, passages):
            return [float(len(p)) for p in passages]  # prefers the longest passage

    index, r = build()
    r.reranker = Reverse()
    ctx = r.retrieve("registros auditoría acceso", "acme", top_k=2)
    assert len(ctx[0].parent.text) >= len(ctx[1].parent.text)


# ---- Qdrant ------------------------------------------------------------------------------

def test_qdrant_store_matches_in_memory_behaviour():
    pytest.importorskip("qdrant_client")
    emb = HashingEmbedder()
    _, mem = build()
    index, qd = build(QdrantVectorStore(emb.dim))
    q = "¿cuánto se conservan los registros de auditoría?"
    assert [c.parent.id for c in qd.retrieve(q, "acme")] == [c.parent.id for c in mem.retrieve(q, "acme")]
    assert qd.retrieve(q, "other") == []


def test_qdrant_acl_delete_and_scroll():
    pytest.importorskip("qdrant_client")
    store = QdrantVectorStore(HashingEmbedder().dim)
    index, r = build(store)
    parents, children = chunk_document("Contrato reservado.", doc_id="legal.md", tenant="acme", source="legal.md", acl=("legal",))
    index.add(parents, children, r.embedder.embed([c.text for c in children]))
    assert r.retrieve("contrato reservado", "acme") == []
    assert r.retrieve("contrato reservado", "acme", frozenset({"legal"}))
    assert {c.doc_id for c in store.iter_children("acme")} == {*DOCS, "legal.md"}
    store.delete_document("acme", "legal.md")
    assert "legal.md" not in {c.doc_id for c in store.iter_children("acme")}
    assert store.health()


def test_qdrant_encrypts_text_at_rest_and_bm25_rebuilds_from_ciphertext_store():
    pytest.importorskip("qdrant_client")
    pytest.importorskip("cryptography")
    from contextrag.core.security import TenantCipher

    store = QdrantVectorStore(HashingEmbedder().dim, cipher=TenantCipher(b"0123456789abcdef-master"))
    build(store)
    raw = store.client.scroll(store.col, limit=50, with_payload=True)[0]
    assert raw and all("auditoría" not in p.payload["text"] for p in raw)
    r = HybridRetriever(HybridIndex(store), HashingEmbedder(), LexicalReranker(), Settings())
    assert r.retrieve("registros de auditoría", "acme")[0].parent.doc_id == "audit.md"


# ---- embedding cache ---------------------------------------------------------------------

def test_cached_embedder_only_computes_misses():
    class Counting(HashingEmbedder):
        calls: list[str] = []

        def embed(self, texts):
            self.calls.extend(texts)
            return super().embed(texts)

    inner = Counting()
    emb = CachedEmbedder(inner, InMemoryCache(), "m")
    a = emb.embed(["uno", "dos"])
    b = emb.embed(["dos", "tres", "uno"])
    assert inner.calls == ["uno", "dos", "tres"]
    assert b[0] == a[1] and b[2] == a[0]
    assert math.isclose(sum(x * x for x in b[1]), 1.0, rel_tol=1e-6)
