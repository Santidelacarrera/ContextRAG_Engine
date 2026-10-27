import base64

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import factory
from contextrag.api.server import create_app
from conftest import POLICY, make_pdf

H = {"X-Tenant": "acme"}


@pytest.fixture
def client():
    return TestClient(create_app(factory.offline()))


def ingest(client, **doc):
    doc.setdefault("doc_id", "policy.md")
    return client.post("/ingest", json={"documents": [doc]}, headers=H)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok", "components": {"index": True, "cache": True, "audit": True}}


def test_healthz_reports_degraded_component_with_503():
    rag = factory.offline()
    rag.cache.health = lambda: False
    r = TestClient(create_app(rag)).get("/healthz")
    assert r.status_code == 503 and r.json()["status"] == "degraded" and r.json()["components"]["cache"] is False


def test_healthz_unavailable_when_backend_cannot_start(monkeypatch):
    monkeypatch.setattr(factory, "from_env", lambda: (_ for _ in ()).throw(RuntimeError("qdrant down")))
    r = TestClient(create_app()).get("/healthz")
    assert r.status_code == 503 and r.json()["error"] == "RuntimeError"
    assert "qdrant down" not in r.text  # the public probe names the class of the failure, never its message


def test_ingest_then_query_returns_audited_answer_with_coordinates(client):
    r = ingest(client, text=POLICY)
    assert r.status_code == 200
    [doc] = r.json()["documents"]
    assert doc["doc_id"] == "policy.md" and doc["kind"] == "markdown" and doc["parents"] >= 1 and doc["children"] >= 1

    r = client.post("/query", json={"question": "¿Cuánto tiempo se conservan los registros de auditoría?"}, headers=H)
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "verified" and j["confidence"] >= 0.95
    ev = j["claims"][0]["evidence"]
    assert ev["doc_id"] == "policy.md" and POLICY[ev["start"] : ev["end"]] == ev["text"] and "7 años" in ev["text"]
    src = j["sources"][0]
    assert src["doc_id"] == "policy.md" and src["start"] <= ev["start"] and ev["end"] <= src["end"]
    assert src["section"] == "Retención de datos"


def test_query_with_unverifiable_answer_returns_deterministic_fallback(client):
    ingest(client, text=POLICY)
    j = client.post("/query", json={"question": "¿Quién ganó el mundial de 1998?"}, headers=H).json()
    assert j["status"] == "fallback"
    assert j["answer"] == "No se puede verificar la respuesta con un 95% de certidumbre basada en las fuentes proporcionadas"


def test_ingest_pdf_base64_reports_pages(client):
    pdf = make_pdf(["Page one is about apples.", "The warranty period is 24 months."])
    r = ingest(client, doc_id="w.pdf", content_base64=base64.b64encode(pdf).decode())
    assert r.status_code == 200 and r.json()["documents"][0]["kind"] == "pdf"
    j = client.post("/query", json={"question": "What is the warranty period?"}, headers=H).json()
    assert j["status"] == "verified"
    assert j["sources"][0]["page"] == 1  # the parent chunk starts on page 1 ...
    assert j["claims"][0]["evidence"]["page"] == 2  # ... but the supporting sentence is on page 2


def test_ingest_code_file_and_multiple_documents(client):
    r = client.post("/ingest", json={"documents": [
        {"doc_id": "a.py", "text": "def limit():\n    # El límite de reintentos es 5.\n    return 5\n"},
        {"doc_id": "b.txt", "text": "Notas varias."},
    ]}, headers=H)
    assert [d["kind"] for d in r.json()["documents"]] == ["code", "text"]


def test_reingesting_a_document_replaces_it(client):
    ingest(client, text="Los registros se conservan durante 7 años.")
    ingest(client, text="Los registros se conservan durante 9 años.")
    j = client.post("/query", json={"question": "¿cuánto se conservan los registros?"}, headers=H).json()
    assert "9 años" in j["answer"] and "7 años" not in j["answer"]


@pytest.mark.parametrize("payload", [
    {"documents": []},
    {"documents": [{"doc_id": "x"}]},  # neither text nor content
    {"documents": [{"doc_id": "x", "text": "a", "content_base64": "YQ=="}]},  # both
    {"documents": [{"doc_id": "x", "content_base64": "%%%not-base64%%%"}]},
    {"documents": [{"doc_id": "x.pdf", "content_base64": base64.b64encode(b"not a pdf").decode()}]},
    {"documents": [{"doc_id": "", "text": "a"}]},
    {"documents": [{"doc_id": "x", "text": "a", "kind": "video"}]},
    {},
])
def test_ingest_validation_errors(client, payload):
    assert client.post("/ingest", json=payload, headers=H).status_code == 422


def test_tenant_header_is_required_and_isolates_data(client):
    ingest(client, text=POLICY)
    assert client.post("/query", json={"question": "x"}).status_code == 400
    assert client.post("/ingest", json={"documents": [{"doc_id": "a", "text": "b"}]}, headers={"X-Tenant": "  "}).status_code == 400
    j = client.post("/query", json={"question": "registros de auditoría"}, headers={"X-Tenant": "rival"}).json()
    assert j["status"] == "fallback" and j["sources"] == []


def test_acl_groups_via_api(client):
    ingest(client, doc_id="s.txt", text="El código secreto es ORIÓN.", acl=["legal"])
    q = {"question": "¿Cuál es el código secreto?"}
    assert client.post("/query", json=q, headers=H).json()["status"] == "fallback"
    assert client.post("/query", json={**q, "groups": ["legal"]}, headers=H).json()["status"] == "verified"


def test_query_validation(client):
    for body in ({}, {"question": ""}, {"question": "x", "top_k": 0}, {"question": "x", "top_k": 99}):
        assert client.post("/query", json=body, headers=H).status_code == 422


def test_repeated_query_is_served_from_cache(client):
    ingest(client, text=POLICY)
    body = {"question": "¿Cuánto tiempo se conservan los registros de auditoría?"}
    assert client.post("/query", json=body, headers=H).json()["cached"] is False
    assert client.post("/query", json=body, headers=H).json()["cached"] is True


def test_audit_endpoint_lists_queries_per_tenant(client):
    ingest(client, text=POLICY)
    client.post("/query", json={"question": "¿Cómo se cifran los datos en reposo?"}, headers={**H, "X-User": "ana"})
    rows = client.get("/audit", headers=H).json()
    assert rows[0]["user"] == "ana" and rows[0]["status"] == "verified"
    assert client.get("/audit", headers={"X-Tenant": "rival"}).json() == []


def test_api_key_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("CONTEXTRAG_API_KEY", "s3cret")
    body = {"documents": [{"doc_id": "a", "text": "b"}]}
    assert client.post("/ingest", json=body, headers=H).status_code == 401
    assert client.post("/ingest", json=body, headers={**H, "X-API-Key": "wrong"}).status_code == 401
    assert client.post("/ingest", json=body, headers={**H, "X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/healthz").status_code == 200  # probes stay open


def test_delete_document_removes_it_from_answers_and_cache(client):
    ingest(client, text=POLICY)
    body = {"question": "¿Cuánto tiempo se conservan los registros de auditoría?"}
    assert client.post("/query", json=body, headers=H).json()["status"] == "verified"
    r = client.delete("/documents/policy.md", headers=H)
    assert r.status_code == 200 and {k: r.json()[k] for k in ("tenant", "deleted")} == {"tenant": "acme", "deleted": "policy.md"}
    j = client.post("/query", json=body, headers=H).json()
    assert j["status"] == "fallback" and not j["cached"]  # stale cached answer must not survive deletion
    assert client.delete("/documents/policy.md", headers=H).status_code == 200  # idempotent
    assert client.delete("/documents/x", headers={}).status_code == 400


def test_delete_is_tenant_scoped_and_supports_nested_ids(client):
    ingest(client, doc_id="docs/a.md", text="El plazo es de 30 días.")
    client.delete("/documents/docs/a.md", headers={"X-Tenant": "rival"})
    assert client.post("/query", json={"question": "¿cuál es el plazo?"}, headers=H).json()["status"] == "verified"
    client.delete("/documents/docs/a.md", headers=H)
    assert client.post("/query", json={"question": "¿cuál es el plazo?"}, headers=H).json()["status"] == "fallback"


def test_size_limits_reject_oversized_requests(client):
    big = {"documents": [{"doc_id": f"d{i}", "text": "x"} for i in range(51)]}
    assert client.post("/ingest", json=big, headers=H).status_code == 422
    assert client.post("/query", json={"question": "x" * 4001}, headers=H).status_code == 422
    assert client.post("/ingest", json={"documents": [{"doc_id": "d" * 513, "text": "x"}]}, headers=H).status_code == 422


def test_metrics_endpoint(client):
    ingest(client, text=POLICY)
    client.post("/query", json={"question": "¿Cuánto tiempo se conservan los registros de auditoría?"}, headers=H)
    r = client.get("/metrics")
    assert r.status_code == 200 and 'contextrag_queries_total{status="verified"} 1' in r.text


def test_ingest_reports_unchanged_and_force_reembeds(client):
    assert ingest(client, text=POLICY).json()["documents"][0]["unchanged"] is False
    assert ingest(client, text=POLICY).json()["documents"][0]["unchanged"] is True
    assert ingest(client, text=POLICY, force=True).json()["documents"][0]["unchanged"] is False


def test_list_documents_is_tenant_scoped_and_shows_acl_docs(client):
    ingest(client, text=POLICY)
    ingest(client, doc_id="secret.txt", text="Plan reservado.", acl=["legal"])
    docs = client.get("/documents", headers=H).json()["documents"]
    assert [d["doc_id"] for d in docs] == ["policy.md", "secret.txt"]
    sec = docs[1]
    assert sec["acl"] == ["legal"] and sec["parents"] == 1 and sec["chars"] == len("Plan reservado.") and len(sec["doc_hash"]) == 64
    assert client.get("/documents", headers={"X-Tenant": "rival"}).json()["documents"] == []
    client.delete("/documents/policy.md", headers=H)
    assert [d["doc_id"] for d in client.get("/documents", headers=H).json()["documents"]] == ["secret.txt"]
    assert client.get("/documents").status_code == 400


# ---- graceful shutdown ---------------------------------------------------------------------

def test_shutdown_closes_what_the_app_built_but_not_what_the_caller_injected(monkeypatch):
    from fastapi.testclient import TestClient

    from contextrag import factory
    from contextrag.api.server import create_app

    closed = []

    class Tracked:
        def __init__(self, name):
            self.name = name

        def health(self):
            return {"index": True}

        def close(self):
            closed.append(self.name)

    monkeypatch.setattr(factory, "from_env", lambda: Tracked("built"))
    with TestClient(create_app()) as client:  # lifespan runs on enter/exit
        assert client.get("/healthz").status_code == 200  # first use builds the pipeline from the environment
    assert closed == ["built"]

    closed.clear()
    with TestClient(create_app(Tracked("injected"))) as client:
        assert client.get("/healthz").status_code == 200
    assert closed == []  # the caller owns an injected pipeline
