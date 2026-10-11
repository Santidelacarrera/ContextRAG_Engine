import json
import time

import pytest

from contextrag import Status, factory
from contextrag.core.chunking import chunk_document
from contextrag.core.embeddings import HashingEmbedder
from contextrag.retrieval.bm25 import BM25Index, IncrementalBM25Index
from contextrag.retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore
from conftest import POLICY

Q = "¿Cuánto tiempo se conservan los registros de auditoría?"


def seeded():
    r = factory.offline()
    for t in ("acme", "rival"):
        r.ingestor.ingest_text(POLICY, tenant=t, doc_id="policy.md")
        r.ingestor.ingest_text("El código secreto es ORIÓN.", tenant=t, doc_id="s.txt", acl=("legal",))
        r.ask(Q, tenant=t, user="u")
    return r


# ---- stores / indexes ----------------------------------------------------------------------

@pytest.mark.parametrize("make", [InMemoryVectorStore, lambda: QdrantVectorStore(HashingEmbedder().dim)])
def test_store_delete_tenant_removes_only_that_tenant(make):
    pytest.importorskip("qdrant_client")
    store, emb = make(), HashingEmbedder()
    for t in ("a", "b"):
        for d in ("x", "y"):
            p, c = chunk_document(f"Contenido {t} {d}.", doc_id=d, tenant=t, source=d)
            store.add(p, c, emb.embed([x.text for x in c]))
    assert store.delete_tenant("a") == 2
    assert store.list_documents("a") == [] and list(store.iter_children("a")) == []
    assert store.dense(emb.embed(["Contenido a x"])[0], "a", frozenset(), 5) == []
    assert [d["doc_id"] for d in store.list_documents("b")] == ["x", "y"] and len(list(store.iter_children("b"))) == 2
    assert store.delete_tenant("a") == 0 and store.delete_tenant("never-existed") == 0


@pytest.mark.parametrize("cls", [BM25Index, IncrementalBM25Index])
def test_bm25_delete_tenant_and_reonboarding(cls):
    from contextrag.core.models import ChildChunk

    store = InMemoryVectorStore()
    emb = HashingEmbedder()
    chunks = [ChildChunk("a1", "p", "d", "a", "manzanas rojas", 0, 1), ChildChunk("b1", "p", "d", "b", "manzanas verdes", 0, 1)]
    store.add([], chunks, emb.embed([c.text for c in chunks]))
    idx = cls(loader=lambda t: store.iter_children(t))
    assert idx.search("manzanas", "a", frozenset(), 5)  # lazily loaded
    store.delete_tenant("a")
    idx.delete_tenant("a")
    assert idx.search("manzanas", "a", frozenset(), 5) == []  # not resurrected from anywhere
    assert [c.id for c, _ in idx.search("manzanas", "b", frozenset(), 5)] == ["b1"]
    idx.add([ChildChunk("a2", "p", "d2", "a", "peras amarillas", 0, 1)])  # the tenant can come back
    assert [c.id for c, _ in idx.search("peras", "a", frozenset(), 5)] == ["a2"]


# ---- pipeline ------------------------------------------------------------------------------

def test_erase_tenant_removes_documents_cache_and_audit_but_not_other_tenants():
    r = seeded()
    assert r.ask(Q, tenant="acme").cached  # a cached answer exists
    out = r.erase_tenant("acme")
    assert out == {"documents": 2, "audit_records": 2}
    assert r.index.list_documents("acme") == [] and r.audit.list("acme") == []
    a = r.ask(Q, tenant="acme", groups=frozenset({"legal"}))
    assert a.status is Status.FALLBACK and not a.cached and a.sources == []  # nothing, not even from the cache
    assert r.ask(Q, tenant="rival").status is Status.VERIFIED  # untouched
    assert len(r.index.list_documents("rival")) == 2 and len(r.audit.list("rival")) >= 1


def test_erase_tenant_legal_hold_keeps_audit_and_is_idempotent():
    r = seeded()
    out = r.erase_tenant("acme", include_audit=False)
    assert out["audit_records"] == 0 and len(r.audit.list("acme")) == 1
    assert r.erase_tenant("acme") == {"documents": 0, "audit_records": 1}
    assert r.erase_tenant("never-existed") == {"documents": 0, "audit_records": 0}
    with pytest.raises(ValueError):
        r.erase_tenant("")


def test_erased_tenant_can_be_reingested_cleanly():
    r = seeded()
    r.erase_tenant("acme")
    info = r.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    assert info["unchanged"] is False  # the idempotency hash went away with the data
    assert r.ask(Q, tenant="acme").status is Status.VERIFIED


def test_audit_retention_purge():
    r = seeded()
    old = r.audit.records[0]
    old["ts"] = time.time() - 40 * 86400
    assert r.purge_audit(30) == 1 and old not in r.audit.records
    assert r.purge_audit(30) == 0
    assert r.purge_audit(0) >= 1 and r.audit.records == []
    with pytest.raises(ValueError):
        r.purge_audit(-1)
    r.audit = None
    assert r.purge_audit(1) == 0 and r.erase_tenant("x") == {"documents": 0, "audit_records": 0}


# ---- API -----------------------------------------------------------------------------------

def test_api_erasure_requires_explicit_confirmation():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from contextrag.api.server import create_app

    c = TestClient(create_app(seeded()))
    h = {"X-Tenant": "acme"}
    assert c.delete("/tenant", headers=h).status_code == 400
    assert c.delete("/tenant", headers={**h, "X-Confirm-Tenant": "rival"}).status_code == 400
    assert len(c.get("/documents", headers=h).json()["documents"]) == 2  # nothing happened yet
    r = c.delete("/tenant", headers={**h, "X-Confirm-Tenant": "acme"})
    assert r.status_code == 200 and r.json() == {"tenant": "acme", "documents": 2, "audit_records": 1}
    assert c.get("/documents", headers=h).json()["documents"] == []
    assert len(c.get("/documents", headers={"X-Tenant": "rival"}).json()["documents"]) == 2
    kept = c.delete("/tenant?keep_audit=true", headers={"X-Tenant": "rival", "X-Confirm-Tenant": "rival"}).json()
    assert kept["audit_records"] == 0 and len(c.get("/audit", headers={"X-Tenant": "rival"}).json()) >= 1


def test_api_erasure_is_admin_only_and_bound_to_the_keys_tenant():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from contextrag.api.auth import KeyStore, new_entry
    from contextrag.api.server import create_app

    entries, k = {}, {}
    for kid, tenant, role in (("ana", "acme", "admin"), ("bob", "acme", "reader"), ("eve", "rival", "admin")):
        key, digest, e = new_entry(kid, tenant, role)
        entries[digest], k[kid] = e, key
    c = TestClient(create_app(seeded(), keystore=KeyStore(entries)))
    confirm = lambda who, t: {"X-API-Key": k[who], "X-Confirm-Tenant": t}
    assert c.delete("/tenant", headers=confirm("bob", "acme")).status_code == 403  # reader
    assert c.delete("/tenant", headers={"X-API-Key": k["ana"]}).status_code == 400  # no confirmation
    # eve (admin of "rival") cannot erase acme: her key's tenant is what counts, so confirming "acme" is wrong
    assert c.delete("/tenant", headers=confirm("eve", "acme")).status_code == 400
    assert c.delete("/tenant", headers={**confirm("eve", "acme"), "X-Tenant": "acme"}).status_code == 403
    assert c.delete("/tenant", headers=confirm("ana", "acme")).json()["documents"] == 2


# ---- CLI -----------------------------------------------------------------------------------

def test_cli_erase_requires_yes(capsys):
    from contextrag.__main__ import main

    assert main(["erase", "--tenant", "acme"]) == 2
    assert main(["erase", "--tenant", "acme", "--yes"]) == 0
    assert json.loads(capsys.readouterr().out) == {"documents": 0, "audit_records": 0}
    assert main(["audit-purge", "--older-than-days", "30"]) == 0
    assert json.loads(capsys.readouterr().out) == {"purged": 0}
