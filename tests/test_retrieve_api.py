"""POST /api/v1/retrieve: contract, auth binding, error mapping, fail-closed audit and real RLS behaviour."""
import contextlib
import json
import logging

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import factory
from contextrag.api.auth import KeyStore, hash_key
from contextrag.api.server import create_app
from conftest import POLICY

URL = "/api/v1/retrieve"
QUESTION = "¿Cuánto tiempo se conservan los registros de auditoría?"
H = {"X-Tenant": "acme"}


class FakeConn:
    pass


class FakeDB:
    """Stands in for contextrag.db.TenantDB so that the HTTP layer can be tested without PostgreSQL."""

    def __init__(self, open_error=None, log_error=None):
        self.open_error, self.log_error = open_error, log_error
        self.opened_for: list[str] = []
        self.records: list[dict] = []

    @contextlib.contextmanager
    def tenant_transaction(self, tenant):
        if self.open_error:
            raise self.open_error
        self.opened_for.append(tenant)
        yield FakeConn()  # committed only if the body completes: a failed write must not leak fragments

    def log_retrieval(self, conn, record):
        if self.log_error:
            raise self.log_error
        self.records.append(record)


@pytest.fixture
def rag():
    r = factory.offline()
    r.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    r.ingestor.ingest_text("Los gatos duermen todo el día en el sofá.", tenant="rival", doc_id="gatos.md")
    return r


def post(client, body, headers=H):
    return client.post(URL, json=body, headers=headers)


# ---- contract -----------------------------------------------------------------------------

def test_returns_fragments_with_metadata_scores_and_an_unverified_draft(rag):
    db = FakeDB()
    r = post(TestClient(create_app(rag, tenant_db=db)), {"tenant_id": "acme", "query": QUESTION, "top_k": 3})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["tenant_id"] == "acme" and j["query"] == QUESTION
    assert 1 <= len(j["fragments"]) <= 3
    f = j["fragments"][0]
    assert f["doc_id"] == "policy.md" and f["section"] == "Retención de datos" and isinstance(f["score"], float)
    assert POLICY[f["start"] : f["end"]] == f["text"]  # coordinates point into the source document
    assert j["draft_answer"] and "7 años" in j["draft_answer"] and j["draft_verified"] is False
    assert j["latency_ms"] >= 0


def test_retrieval_is_recorded_inside_the_callers_tenant_transaction(rag):
    db = FakeDB()
    j = post(TestClient(create_app(rag, tenant_db=db)), {"tenant_id": "acme", "query": QUESTION, "top_k": 2}).json()
    assert db.opened_for == ["acme"]
    [rec] = db.records
    assert rec["tenant"] == "acme" and rec["query"] == QUESTION and rec["top_k"] == 2
    assert rec["fragments"] == len(j["fragments"]) and rec["doc_ids"] == ["policy.md"] * len(j["fragments"])


def test_empty_tenant_returns_ok_envelope_with_no_fragments_and_no_draft(rag):
    j = post(TestClient(create_app(rag, tenant_db=FakeDB())), {"tenant_id": "vacio", "query": QUESTION},
             headers={"X-Tenant": "vacio"}).json()
    assert j["status"] == "no_results" and j["fragments"] == [] and j["draft_answer"] is None


def test_search_never_returns_another_tenants_chunks(rag):
    j = post(TestClient(create_app(rag, tenant_db=FakeDB())), {"tenant_id": "acme", "query": "gatos duermen sofá"}).json()
    assert all(f["doc_id"] == "policy.md" for f in j["fragments"])
    assert not any("gatos" in f["text"] for f in j["fragments"])


@pytest.mark.parametrize("body", [
    {"tenant_id": "acme", "query": ""},                                  # empty query
    {"tenant_id": "acme", "query": "x" * 4001},                          # too long
    {"tenant_id": "acme", "query": "q", "top_k": 0},                     # below range
    {"tenant_id": "acme", "query": "q", "top_k": 21},                    # above range
    {"tenant_id": "acme; DROP TABLE x", "query": "q"},                   # tenant pattern
    {"tenant_id": "", "query": "q"},
    {"tenant_id": "acme", "query": "q", "tenant": "rival"},              # unknown field is refused, not ignored
    {"query": "q"},                                                      # missing tenant_id
])
def test_invalid_payloads_are_rejected_with_422(rag, body):
    assert post(TestClient(create_app(rag, tenant_db=FakeDB())), body).status_code == 422


# ---- tenant binding ---------------------------------------------------------------------

def test_body_tenant_must_match_the_header_tenant(rag):
    db = FakeDB()
    r = post(TestClient(create_app(rag, tenant_db=db)), {"tenant_id": "rival", "query": QUESTION})
    assert r.status_code == 403 and db.opened_for == [] and db.records == []  # refused before touching the store


def test_api_key_binds_the_tenant_and_conflicting_claims_are_refused(rag):
    keys = KeyStore({hash_key("k-acme"): {"id": "ana", "tenant": "acme", "groups": [], "role": "reader"}})
    c = TestClient(create_app(rag, keystore=keys, tenant_db=FakeDB()))
    key = {"X-API-Key": "k-acme"}
    assert post(c, {"tenant_id": "acme", "query": QUESTION}, headers=key).status_code == 200
    assert post(c, {"tenant_id": "rival", "query": QUESTION}, headers=key).status_code == 403
    assert post(c, {"tenant_id": "acme", "query": QUESTION}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert post(c, {"tenant_id": "acme", "query": QUESTION}, headers={}).status_code == 401


# ---- error mapping ----------------------------------------------------------------------

def test_unreachable_database_returns_503_with_retry_after_and_no_internals(rag):
    import psycopg

    c = TestClient(create_app(rag, tenant_db=FakeDB(open_error=psycopg.OperationalError("connection refused 10.0.0.5"))))
    r = post(c, {"tenant_id": "acme", "query": QUESTION})
    assert r.status_code == 503 and r.headers["Retry-After"] == "5"
    assert r.json()["error"] == "database_unavailable" and "10.0.0.5" not in r.text


def test_exhausted_or_closed_pool_returns_503(rag):
    from psycopg_pool import PoolClosed, PoolTimeout

    for exc in (PoolTimeout("timeout"), PoolClosed("closed")):
        r = post(TestClient(create_app(rag, tenant_db=FakeDB(open_error=exc))), {"tenant_id": "acme", "query": QUESTION})
        assert r.status_code == 503 and r.json()["error"] == "database_unavailable"


def test_row_level_security_denial_returns_403(rag):
    import psycopg

    c = TestClient(create_app(rag, tenant_db=FakeDB(open_error=psycopg.errors.InsufficientPrivilege("new row violates RLS"))))
    r = post(c, {"tenant_id": "acme", "query": QUESTION})
    assert r.status_code == 403 and r.json()["error"] == "tenant_not_authorized"


def test_other_database_errors_return_500_without_leaking_sql(rag):
    import psycopg

    c = TestClient(create_app(rag, tenant_db=FakeDB(open_error=psycopg.DatabaseError("syntax near SECRET_COLUMN"))))
    r = post(c, {"tenant_id": "acme", "query": QUESTION})
    assert r.status_code == 500 and r.json()["error"] == "database_error" and "SECRET_COLUMN" not in r.text


def test_failed_audit_write_fails_closed_and_returns_no_fragments(rag):
    import psycopg

    c = TestClient(create_app(rag, tenant_db=FakeDB(log_error=psycopg.OperationalError("lost connection"))))
    r = post(c, {"tenant_id": "acme", "query": QUESTION})
    assert r.status_code == 503 and "fragments" not in r.json() and "7 años" not in r.text


def test_without_a_database_the_endpoint_is_unavailable_not_silently_unaudited(rag, monkeypatch):
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    r = post(TestClient(create_app(rag)), {"tenant_id": "acme", "query": QUESTION})
    assert r.status_code == 503 and r.json()["error"] == "retrieval_store_unavailable"


# ---- logging ----------------------------------------------------------------------------

def events(caplog, name):
    """Structured events emitted by the retrieval route, parsed from their JSON log lines."""
    out = []
    for r in caplog.records:
        if r.name == "contextrag.retrieve" and r.getMessage().startswith("{"):
            e = json.loads(r.getMessage())
            if e["event"] == name:
                out.append((r, e))
    return out


def test_each_served_request_emits_a_structured_event_with_its_tenant(rag, caplog):
    caplog.set_level(logging.INFO, logger="contextrag.retrieve")
    r = post(TestClient(create_app(rag, tenant_db=FakeDB())), {"tenant_id": "acme", "query": QUESTION})
    [(_, e)] = events(caplog, "retrieve.served")
    assert e["tenant_id"] == "acme" and e["status"] == "ok" and e["fragments"] >= 1
    assert e["request_id"] == r.headers["X-Request-Id"] and e["latency_ms"] >= 0


def test_refused_requests_are_logged_with_claimed_and_authenticated_tenant_and_reason(rag, caplog):
    caplog.set_level(logging.WARNING, logger="contextrag.retrieve")
    r = post(TestClient(create_app(rag, tenant_db=FakeDB())), {"tenant_id": "rival", "query": QUESTION})
    assert r.status_code == 403 and r.json()["error"] == "tenant_mismatch"
    [(rec, e)] = events(caplog, "retrieve.rejected")
    assert rec.levelno == logging.WARNING
    assert e["tenant_id"] == "acme" and e["claimed_tenant_id"] == "rival" and e["reason"] == "tenant_mismatch"
    assert e["http_status"] == 403 and e["request_id"] == r.headers["X-Request-Id"]


def test_rls_denials_are_logged_as_403_events_with_the_tenant(rag, caplog):
    import psycopg

    caplog.set_level(logging.WARNING, logger="contextrag.retrieve")
    c = TestClient(create_app(rag, tenant_db=FakeDB(open_error=psycopg.errors.InsufficientPrivilege("RLS"))))
    post(c, {"tenant_id": "acme", "query": QUESTION})
    [(_, e)] = events(caplog, "retrieve.rejected")
    assert e["reason"] == "rls_denied" and e["http_status"] == 403 and e["tenant_id"] == "acme"
    assert e["error_class"] == "InsufficientPrivilege"


def test_pool_failures_are_logged_as_503_events_without_connection_details(rag, caplog):
    import psycopg

    caplog.set_level(logging.ERROR, logger="contextrag.retrieve")
    c = TestClient(create_app(rag, tenant_db=FakeDB(open_error=psycopg.OperationalError("password=hunter2 host=db"))))
    r = post(c, {"tenant_id": "acme", "query": QUESTION})
    [(rec, e)] = events(caplog, "retrieve.failed")
    assert rec.levelno == logging.ERROR and e["reason"] == "database_unavailable" and e["http_status"] == 503
    assert e["tenant_id"] == "acme" and e["error_class"] == "OperationalError"
    assert "hunter2" not in rec.getMessage() and r.headers["Retry-After"] == "5"


def test_metrics_count_rejections_and_failures_by_reason(rag):
    import psycopg

    c = TestClient(create_app(rag, tenant_db=FakeDB(open_error=psycopg.errors.InsufficientPrivilege("RLS"))))
    post(c, {"tenant_id": "rival", "query": QUESTION})  # tenant_mismatch
    post(c, {"tenant_id": "acme", "query": QUESTION})   # rls_denied
    text = rag.metrics_text()
    assert 'contextrag_retrieve_rejected_total{reason="tenant_mismatch"} 1' in text
    assert 'contextrag_retrieve_rejected_total{reason="rls_denied"} 1' in text


# ---- real PostgreSQL: RLS on the transaction and on the audit table -----------------------

@pytest.fixture(scope="module")
def tenant_db(pg_dsn, pg_app_dsn):
    from contextrag.db import TenantDB

    db = TenantDB(pg_app_dsn, admin_dsn=pg_dsn)
    yield db
    db.close()


def test_real_database_stores_the_audit_row_only_for_the_caller(tenant_db, rag, pg_dsn, pg_app_dsn):
    import psycopg

    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        admin.execute("DELETE FROM retrieval_audit")
    c = TestClient(create_app(rag, tenant_db=tenant_db))
    assert post(c, {"tenant_id": "acme", "query": QUESTION}).status_code == 200
    assert post(c, {"tenant_id": "rival", "query": "gatos sofá"}, headers={"X-Tenant": "rival"}).status_code == 200

    with psycopg.connect(pg_app_dsn, autocommit=True) as raw:
        assert raw.execute("SELECT count(*) FROM retrieval_audit").fetchone()[0] == 0  # no tenant set: invisible
        for tenant, expected in (("acme", [QUESTION]), ("rival", ["gatos sofá"])):
            with raw.transaction():
                raw.execute("SELECT set_config('app.tenant', %s, true)", (tenant,))
                assert [q for (q,) in raw.execute("SELECT query FROM retrieval_audit").fetchall()] == expected


def test_app_role_cannot_forge_other_tenants_or_tamper_with_the_audit(tenant_db, pg_dsn, pg_app_dsn):
    import psycopg

    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # WITH CHECK: no writes as another tenant
        with tenant_db.tenant_transaction("acme") as conn:
            tenant_db.log_retrieval(conn, {"id": "00000000-0000-0000-0000-000000000001", "tenant": "rival",
                                           "user": None, "query": "forged", "top_k": 1, "fragments": 0,
                                           "doc_ids": [], "latency_ms": 0.0})
    with psycopg.connect(pg_app_dsn, autocommit=True) as raw:  # append-only: no UPDATE or DELETE grant
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with raw.transaction():
                raw.execute("SELECT set_config('app.tenant', 'acme', true)")
                raw.execute("DELETE FROM retrieval_audit")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            raw.execute("DROP TABLE retrieval_audit")


def test_endpoint_returns_503_when_the_database_is_shut_down(rag, pg_dsn, pg_app_dsn):
    from contextrag.db import TenantDB

    db = TenantDB(pg_app_dsn, admin_dsn=pg_dsn)
    db.close()  # the pool is gone: every request must fail closed
    r = post(TestClient(create_app(rag, tenant_db=db)), {"tenant_id": "acme", "query": QUESTION})
    assert r.status_code == 503 and r.json()["error"] == "database_unavailable"
    assert db.health() is False
