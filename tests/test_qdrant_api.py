import pytest

pytest.importorskip("qdrant_client")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import ContextRAG, Status
from contextrag.api import create_app
from contextrag.audit import InMemoryAudit
from contextrag.embeddings import HashingEmbedder
from contextrag.llm import ExtractiveLLM
from contextrag.nli import LexicalEntailment
from contextrag.qdrant_store import QdrantStore
from contextrag.rerank import LexicalReranker
from contextrag.security import TenantCipher

DOC = "Los registros de auditoría se conservan durante 7 años.\n\nLas claves se rotan cada 12 meses."


def make(cipher=None):
    emb = HashingEmbedder()
    return ContextRAG(store=QdrantStore(emb.dim, cipher=cipher), embedder=emb, reranker=LexicalReranker(),
                      llm=ExtractiveLLM(), nli=LexicalEntailment(), audit=InMemoryAudit())


def test_qdrant_hybrid_with_encryption_and_isolation():
    pytest.importorskip("cryptography")
    r = make(TenantCipher(b"0123456789abcdef-master"))
    r.ingestor.ingest_text(DOC, tenant="acme", doc_id="d")
    a = r.ask("¿Cuánto se conservan los registros de auditoría?", tenant="acme")
    assert a.status is Status.VERIFIED and "7 años" in a.claims[0].evidence.text
    raw = r.retriever.store.client.scroll(r.retriever.store.col, with_payload=True, limit=10)[0]
    assert all("auditoría" not in p.payload["text"] for p in raw)  # ciphertext at rest
    assert r.ask("registros de auditoría", tenant="other").status is Status.FALLBACK


def test_api_roundtrip():
    c = TestClient(create_app(make()))
    h = {"X-Tenant": "acme"}
    assert c.post("/ingest", json={"doc_id": "d", "text": DOC}, headers=h).json()["chunks"] >= 1
    j = c.post("/ask", json={"question": "¿Cuándo se rotan las claves?"}, headers=h).json()
    assert j["status"] == "verified" and j["claims"][0]["evidence"]["start"] >= 0
    assert c.get("/audit", headers=h).json()[0]["question"].startswith("¿Cuándo")
    assert c.post("/ask", json={"question": "x"}).status_code == 400
