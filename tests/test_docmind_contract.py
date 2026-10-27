"""Contract tests between the reference client (contextrag.client) and a real ContextRAG server on a local port.

Real HTTP through uvicorn, real API keys, synthetic documents, the offline stack. These check the contract that a consumer
such as DOCMIND_AI would rely on. They do not exercise DOCMIND_AI, PostgreSQL, Qdrant or a model provider."""
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

uvicorn = pytest.importorskip("uvicorn")

from contextrag import factory  # noqa: E402
from contextrag.api.auth import KeyStore, new_entry  # noqa: E402
from contextrag.api.server import create_app  # noqa: E402
from contextrag.client import ContextRAGClient, ContextRAGError  # noqa: E402

ALFA = "El expediente ALFA fija la audiencia preliminar el 5 de mayo de 2026 en el Juzgado Tercero."
ALFA_NEW = "El expediente ALFA fija la audiencia preliminar el 21 de octubre de 2026 en el Juzgado Cuarto."
Q = "¿Cuándo es la audiencia preliminar del expediente ALFA?"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    specs = [("svc", "org-1", "admin", ()), ("reader-a", "org-1", "reader", ("caso-a",)),
             ("reader-b", "org-1", "reader", ("caso-b",)), ("other", "org-2", "admin", ())]
    entries, plain = {}, {}
    for kid, tenant, role, groups in specs:
        key, digest, entry = new_entry(kid, tenant, role, groups)
        entries[digest], plain[kid] = entry, key
    rag = factory.offline()
    port = _free_port()
    config = uvicorn.Config(create_app(rag, keystore=KeyStore(entries)), host="127.0.0.1", port=port, log_level="warning")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    assert srv.started, "uvicorn did not start"
    base = f"http://127.0.0.1:{port}"
    yield {"base": base, "plain": plain, "rag": rag}
    srv.should_exit = True
    thread.join(timeout=10)
    rag.close()


def client(srv, who):
    return ContextRAGClient(srv["base"], srv["plain"][who], timeout_seconds=10)


# ---- success paths ----------------------------------------------------------------------------------------------

def test_health_is_open_and_reports_components(server):
    body = client(server, "svc").health()
    assert body["status"] in ("ok", "degraded") and "components" in body


def test_ingest_returns_the_tenant_of_the_key_and_marks_unchanged_documents(server):
    svc = client(server, "svc")
    out = svc.ingest([{"doc_id": "alfa.md", "text": ALFA, "source": "alfa.md", "acl": ["caso-a"]}])
    assert out["tenant"] == "org-1"
    assert out["documents"][0]["unchanged"] is False
    again = svc.ingest([{"doc_id": "alfa.md", "text": ALFA, "source": "alfa.md", "acl": ["caso-a"]}])
    assert again["documents"][0]["unchanged"] is True


def test_query_returns_an_answer_with_the_document_and_source_it_came_from(server):
    svc = client(server, "svc")
    svc.ingest([{"doc_id": "alfa.md", "text": ALFA, "source": "alfa.md", "acl": []}])
    body = svc.query(Q)
    assert body["status"] == "verified" and "5 de mayo" in body["answer"]
    assert body["sources"][0]["doc_id"] == "alfa.md" and body["sources"][0]["source"] == "alfa.md"
    supported = [c for c in body["claims"] if c["supported"]]
    assert supported and supported[0]["evidence"]["doc_id"] == "alfa.md"


def test_update_by_resending_the_same_doc_id_replaces_what_is_answered(server):
    svc = client(server, "svc")
    svc.ingest([{"doc_id": "alfa.md", "text": ALFA, "acl": []}])
    assert "5 de mayo" in svc.query(Q)["answer"]
    svc.ingest([{"doc_id": "alfa.md", "text": ALFA_NEW, "acl": []}])
    after = svc.query(Q)
    assert "21 de octubre" in after["answer"] and "5 de mayo" not in after["answer"]


def test_delete_removes_the_document_from_answers_and_from_the_listing(server):
    svc = client(server, "svc")
    svc.ingest([{"doc_id": "alfa.md", "text": ALFA, "acl": []}])
    assert svc.delete_document("alfa.md")["deleted"] == "alfa.md"
    after = svc.query(Q)
    assert after["status"] != "verified" and after["sources"] == []
    assert all(d["doc_id"] != "alfa.md" for d in svc.list_documents()["documents"])


def test_a_document_id_with_a_slash_round_trips_through_the_delete_path(server):
    svc = client(server, "svc")
    svc.ingest([{"doc_id": "casos/2026/alfa.md", "text": ALFA, "acl": []}])
    assert svc.delete_document("casos/2026/alfa.md")["deleted"] == "casos/2026/alfa.md"
    assert all(d["doc_id"] != "casos/2026/alfa.md" for d in svc.list_documents()["documents"])


# ---- isolation ----------------------------------------------------------------------------------------------------

def test_another_organisation_gets_no_source_and_its_delete_does_not_touch_this_one(server):
    client(server, "svc").ingest([{"doc_id": "alfa.md", "text": ALFA, "acl": []}])
    other = client(server, "other")
    assert other.query(Q)["sources"] == []
    other.delete_document("alfa.md")  # the other tenant has no such document: nothing of org-1 is removed
    assert client(server, "svc").query(Q)["status"] == "verified"


def test_a_reader_without_the_case_group_gets_no_text_and_cannot_widen_its_groups(server):
    client(server, "svc").ingest([{"doc_id": "alfa.md", "text": ALFA, "acl": ["caso-a"]}])
    reader_b = client(server, "reader-b")
    body = reader_b.query(Q)
    assert body["sources"] == [] and "5 de mayo" not in body["answer"]
    with pytest.raises(ContextRAGError) as info:
        reader_b.query(Q, groups=["caso-a"])  # asking for a group the key does not hold
    assert info.value.status == 403


def test_a_reader_cannot_ingest_or_delete_and_gets_403(server):
    reader = client(server, "reader-a")
    with pytest.raises(ContextRAGError) as info:
        reader.ingest([{"doc_id": "x.md", "text": ALFA, "acl": []}])
    assert info.value.status == 403
    with pytest.raises(ContextRAGError) as info:
        reader.delete_document("alfa.md")
    assert info.value.status == 403


# ---- errors --------------------------------------------------------------------------------------------------------

def test_a_wrong_key_is_401_and_the_key_is_not_echoed(server):
    bad = ContextRAGClient(server["base"], "crag_not_a_real_key_000000000000", timeout_seconds=10)
    with pytest.raises(ContextRAGError) as info:
        bad.query(Q)
    assert info.value.status == 401
    assert "crag_not_a_real_key" not in str(info.value) and "crag_not_a_real_key" not in repr(bad)


def test_an_unreachable_server_is_a_transport_error_without_a_status(server):
    port = _free_port()  # nothing listens here
    dead = ContextRAGClient(f"http://127.0.0.1:{port}", "crag_x", timeout_seconds=2)
    with pytest.raises(ContextRAGError) as info:
        dead.health()
    assert info.value.status is None and info.value.code


def test_a_client_side_batch_limit_fails_before_any_request():
    c = ContextRAGClient("http://127.0.0.1:9", "crag_x")
    with pytest.raises(ValueError):
        c.ingest([{"doc_id": f"d{i}.md", "text": "x"} for i in range(51)])
    with pytest.raises(ValueError):
        c.ingest([{"doc_id": "d.md", "text": "x", "content_base64": "eA=="}])  # both forms
    with pytest.raises(ValueError):
        c.ingest([{"doc_id": "d.md"}])  # neither form


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://x.example", "http://user:pw@x.example",
                                 "http://x.example/?q=1", "", "localhost:8000"])
def test_the_client_refuses_base_urls_outside_the_contract(url):
    with pytest.raises(ValueError):
        ContextRAGClient(url, "crag_x")


def test_a_redirect_is_not_followed_by_the_client():
    hits = []

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Redirect)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        c = ContextRAGClient(f"http://127.0.0.1:{httpd.server_address[1]}", "crag_x", timeout_seconds=5)
        with pytest.raises(ContextRAGError) as info:
            c.list_documents()
        assert info.value.status == 302
        assert hits == ["/documents"]  # the redirect target was never requested
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---- audit ----------------------------------------------------------------------------------------------------------

def test_the_audit_of_a_query_made_through_the_contract_holds_no_document_text(server):
    rag = server["rag"]
    import dataclasses
    rag.cfg = dataclasses.replace(rag.cfg, audit_question="omit", audit_answer="omit")
    svc = client(server, "svc")
    svc.ingest([{"doc_id": "alfa.md", "text": ALFA, "acl": []}])
    svc.query(Q)
    rows = rag.audit.list("org-1")
    assert rows, "the query must be audited"
    dumped = repr(rows)
    assert "5 de mayo" not in dumped and "Juzgado Tercero" not in dumped and "audiencia" not in dumped


# ---- revocation and audit through the contract -----------------------------------------------------------------------

def test_a_key_removed_from_the_keys_file_stops_working_on_the_next_call(tmp_path):
    import json
    import os

    key_a, digest_a, entry_a = new_entry("svc", "org-1", "admin", ())
    key_b, digest_b, entry_b = new_entry("keep", "org-1", "admin", ())
    path = tmp_path / "keys.json"
    path.write_text(json.dumps({digest_a: entry_a, digest_b: entry_b}))
    rag = factory.offline()
    port = _free_port()
    app = create_app(rag, keystore=KeyStore.from_file(str(path)))  # the same loader the server uses at startup
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1",
                                        port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if srv.started:
                break
            time.sleep(0.02)
        base = f"http://127.0.0.1:{port}"
        assert ContextRAGClient(base, key_a, timeout_seconds=10).list_documents()["tenant"] == "org-1"
        path.write_text(json.dumps({digest_b: entry_b}))  # revoke "svc"
        os.utime(path, (time.time() + 5, time.time() + 5))  # make the change visible to the mtime check
        with pytest.raises(ContextRAGError) as info:
            ContextRAGClient(base, key_a, timeout_seconds=10).list_documents()
        assert info.value.status == 401
        assert ContextRAGClient(base, key_b, timeout_seconds=10).list_documents()["tenant"] == "org-1"
    finally:
        srv.should_exit = True
        thread.join(timeout=10)
        rag.close()


def test_the_audit_endpoint_returns_only_the_callers_tenant(server):
    client(server, "svc").ingest([{"doc_id": "alfa.md", "text": ALFA, "acl": []}])
    client(server, "svc").query(Q)
    mine = client(server, "svc").audit()
    assert mine and all(r["tenant"] == "org-1" for r in mine)
    assert client(server, "other").audit() == []


def test_deleting_twice_is_idempotent_and_a_missing_document_is_not_an_error(server):
    svc = client(server, "svc")
    svc.ingest([{"doc_id": "alfa.md", "text": ALFA, "acl": []}])
    first = svc.delete_document("alfa.md")
    second = svc.delete_document("alfa.md")  # already gone: same success shape, no 404
    assert first["deleted"] == second["deleted"] == "alfa.md"
    assert svc.query(Q)["status"] != "verified"
