"""Contract tests for the DOCMIND reference adapter (contextrag/integrations/docmind.py) against a real ContextRAG server.

Real HTTP (uvicorn), a real keys file loaded the way production loads it, revocation and expiry by editing that file,
synthetic documents, and the offline stack. No DOCMIND_AI code runs here: the directory below stands in for DOCMIND's own
lookup, which is exactly the part DOCMIND must implement and verify."""
import dataclasses
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

uvicorn = pytest.importorskip("uvicorn")

from contextrag import factory  # noqa: E402
from contextrag.api.auth import KeyStore, new_entry  # noqa: E402
from contextrag.api.server import create_app  # noqa: E402
from contextrag.client import ContextRAGError  # noqa: E402
from contextrag.integrations.docmind import (AccessDenied, Credential, DocmindRagAdapter,  # noqa: E402
                                             IdentityMismatch, VerifiedPrincipal)

ORG1, ORG2 = "00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"
USER1, USER2, USER3 = "user-1", "user-2", "user-3"
ALFA = "El expediente ALFA fija la audiencia preliminar el 5 de mayo de 2026 en el Juzgado Tercero."
BRAVO = "El expediente BRAVO fija la audiencia preliminar el 9 de junio de 2026 en el Juzgado Séptimo."
Q_ALFA = "¿Cuándo es la audiencia preliminar del expediente ALFA?"
Q_BRAVO = "¿Cuándo es la audiencia preliminar del expediente BRAVO?"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Env:
    """A running server, its keys file, and the credentials DOCMIND's directory would hold."""

    def __init__(self, path, rag, srv, thread, base, creds, entries):
        self.path, self.rag, self.srv, self.thread, self.base = path, rag, srv, thread, base
        self.creds, self.entries = creds, entries

    def write_keys(self, entries: dict) -> None:
        self.path.write_text(json.dumps(entries))
        stamp = time.time() + 5 + len(str(entries))  # a later mtime, so the server reloads on its next call
        os.utime(self.path, (stamp, stamp))

    def directory(self, mapping=None):
        return _Directory(self.creds if mapping is None else mapping)

    def adapter(self, mapping=None):
        return DocmindRagAdapter(self.base, self.directory(mapping), timeout_seconds=10)


class _Directory:
    """Stands in for DOCMIND's lookup: (organisation, user) -> credential, or nothing."""

    def __init__(self, mapping):
        self.mapping = mapping

    def credential_for(self, who: VerifiedPrincipal):
        return self.mapping.get((who.organization_id, who.user_id))


@pytest.fixture
def env(tmp_path):
    specs = {
        "docmind-u1": (ORG1, "admin", ("caso-a",), USER1),  # uploads and deletes its own case (restricted admin)
        "docmind-u2": (ORG1, "reader", ("caso-b",), USER2),
        "docmind-u3": (ORG1, "admin", (), USER3),  # tenant-wide admin of ORG1
        "docmind-other": (ORG2, "admin", (), "user-x"),
    }
    entries, plain = {}, {}
    for kid, (org, role, groups, _) in specs.items():
        key, digest, entry = new_entry(kid, org, role, groups)
        entries[digest], plain[kid] = entry, key
    path = tmp_path / "keys.json"
    path.write_text(json.dumps(entries))
    rag = factory.offline()
    port = _free_port()
    app = create_app(rag, keystore=KeyStore.from_file(str(path)))
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    assert srv.started
    creds = {(org, user): Credential(plain[kid], kid) for kid, (org, _, _, user) in specs.items()}
    e = Env(path, rag, srv, thread, f"http://127.0.0.1:{port}", creds, entries)
    yield e
    srv.should_exit = True
    thread.join(timeout=10)
    rag.close()


def who(org=ORG1, user=USER1):
    return VerifiedPrincipal(organization_id=org, user_id=user)


# ---- successful flow ----------------------------------------------------------------------------------------------

def test_ingest_query_and_delete_run_under_the_users_own_credential(env):
    a = env.adapter()
    out = a.ingest(who(), [{"doc_id": "alfa.md", "text": ALFA, "source": "alfa.md", "acl": ["caso-a"]}])
    assert out["tenant"] == ORG1
    body = a.query(who(), Q_ALFA)
    assert body["status"] == "verified" and "5 de mayo" in body["answer"]
    assert body["sources"][0]["doc_id"] == "alfa.md"
    assert a.delete_document(who(), "alfa.md")["deleted"] == "alfa.md"
    assert a.query(who(), Q_ALFA)["sources"] == []


def test_the_adapter_reports_the_identity_the_credential_has(env):
    assert env.adapter().list_documents(who())["tenant"] == ORG1
    client_me = env.adapter()._client(who()).me()
    assert client_me == {"key_id": "docmind-u1", "tenant": ORG1, "role": "admin", "groups": ["caso-a"]}


# ---- authorisation and isolation between users and organisations ------------------------------------------------

def test_a_user_without_a_credential_is_denied_before_any_request_is_sent(env):
    dead = DocmindRagAdapter("http://127.0.0.1:9", _Directory({}), timeout_seconds=2)  # nothing listens here
    with pytest.raises(AccessDenied):
        dead.query(who(user="nobody"), Q_ALFA)


def test_a_credential_of_another_organisation_is_refused_and_nothing_is_written(env):
    other_key = env.creds[(ORG2, "user-x")]
    wrong = {(ORG1, USER1): other_key}  # DOCMIND's table is wrong: it hands org-2's key to an org-1 user
    with pytest.raises(IdentityMismatch):
        env.adapter(wrong).ingest(who(), [{"doc_id": "leak.md", "text": ALFA, "acl": []}])
    assert env.rag.index.list_documents(ORG2) == []  # the other organisation received nothing
    assert all(d["doc_id"] != "leak.md" for d in env.rag.index.list_documents(ORG1))


def test_a_credential_recorded_for_a_different_key_is_refused(env):
    mismatched = Credential(env.creds[(ORG1, USER2)].key, "docmind-u1")  # right key, wrong recorded key_id
    with pytest.raises(IdentityMismatch):
        env.adapter({(ORG1, USER2): mismatched}).query(who(user=USER2), Q_ALFA)


def test_one_user_cannot_read_another_users_case_even_in_the_same_organisation(env):
    env.adapter().ingest(who(user=USER1), [{"doc_id": "alfa.md", "text": ALFA, "acl": ["caso-a"]}])
    assert env.adapter().query(who(user=USER1), Q_ALFA)["sources"]
    body = env.adapter().query(who(user=USER2), Q_ALFA)  # holds caso-b only
    assert body["sources"] == [] and "5 de mayo" not in body["answer"]


def test_a_user_cannot_widen_the_groups_of_its_credential(env):
    a = env.adapter()
    with pytest.raises(ContextRAGError) as info:
        a.query(who(user=USER2), Q_ALFA, groups=["caso-a"])
    assert info.value.status == 403


# ---- revocation ------------------------------------------------------------------------------------------------------

def test_a_revoked_credential_stops_working_on_the_next_call(env):
    a = env.adapter()
    a.list_documents(who())
    kept = {d: e for d, e in env.entries.items() if e["id"] != "docmind-u1"}
    env.write_keys(kept)
    with pytest.raises(ContextRAGError) as info:
        a.list_documents(who())
    assert info.value.status == 401  # passed on, not hidden behind a generic error


def test_an_expired_credential_is_refused(env):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    key, digest, entry = new_entry("docmind-u1", ORG1, "reader", ("caso-a",), expires_at=past)
    env.write_keys({digest: entry, **{d: e for d, e in env.entries.items() if e["id"] != "docmind-u1"}})
    with pytest.raises(ContextRAGError) as info:
        env.adapter({(ORG1, USER1): Credential(key, "docmind-u1")}).query(who(), Q_ALFA)
    assert info.value.status == 401


def test_a_permission_revoked_by_narrowing_the_groups_takes_effect_at_once(env):
    env.adapter().ingest(who(user=USER1), [{"doc_id": "alfa.md", "text": ALFA, "acl": ["caso-a"]}])
    assert env.adapter().query(who(user=USER1), Q_ALFA)["sources"]
    entries = dict(env.entries)
    digest = next(d for d, e in entries.items() if e["id"] == "docmind-u1")
    entries[digest] = {**entries[digest], "groups": []}  # the key keeps existing but loses caso-a
    env.write_keys(entries)
    assert env.adapter().query(who(user=USER1), Q_ALFA)["sources"] == []


def test_a_missing_or_malformed_key_is_401(env):
    for header in ({}, {"X-API-Key": "crag_wrong"}):
        req = urllib.request.Request(env.base + "/me", headers=header)
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(req, timeout=5)
        assert info.value.code == 401


# ---- deletion and audit attribution -----------------------------------------------------------------------------

def test_deleting_twice_is_idempotent_for_a_tenant_wide_key(env):
    admin = env.adapter()
    admin.ingest(who(user=USER3), [{"doc_id": "alfa.md", "text": ALFA, "acl": []}])
    assert admin.delete_document(who(user=USER3), "alfa.md")["deleted"] == "alfa.md"
    assert admin.delete_document(who(user=USER3), "alfa.md")["deleted"] == "alfa.md"  # already gone: same success


def test_a_case_key_gets_404_for_an_absent_document_so_retries_must_treat_it_as_gone(env):
    # By design a case-restricted key sees the same 404 for "absent" and "belongs to another case": it does not reveal
    # which documents exist. A DOCMIND retry that gets this 404 must read it as "already deleted", not as an error.
    a = env.adapter()
    a.ingest(who(user=USER1), [{"doc_id": "alfa.md", "text": ALFA, "acl": ["caso-a"]}])
    assert a.delete_document(who(user=USER1), "alfa.md")["deleted"] == "alfa.md"
    with pytest.raises(ContextRAGError) as info:
        a.delete_document(who(user=USER1), "alfa.md")
    assert info.value.status == 404


def test_the_audit_attributes_each_query_to_the_credential_user(env):
    env.adapter().ingest(who(user=USER1), [{"doc_id": "alfa.md", "text": ALFA, "acl": ["caso-a"]}])
    env.adapter().query(who(user=USER1), Q_ALFA)
    rows = env.adapter().audit(who(user=USER3))  # tenant-wide admin of the same organisation reads the trail
    users = {r["user"] for r in rows}
    assert users == {"docmind-u1"}  # the key id, which DOCMIND maps to user-1 when it issued the key


def test_a_client_header_cannot_change_the_audited_identity(env):
    # A bound key ignores X-User: another identity sent by the client must not appear in the audit row.
    req = urllib.request.Request(
        env.base + "/query", method="POST",
        data=json.dumps({"question": Q_ALFA}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": env.creds[(ORG1, USER1)].key,
                 "X-User": "someone-else", "X-Tenant": ORG1})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert r.status == 200
    rows = env.adapter().audit(who(user=USER3))
    assert rows and all(r["user"] == "docmind-u1" for r in rows)
    assert all(r["tenant"] == ORG1 for r in rows)


def test_a_client_header_naming_another_organisation_is_refused(env):
    req = urllib.request.Request(
        env.base + "/query", method="POST", data=json.dumps({"question": Q_ALFA}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": env.creds[(ORG1, USER1)].key, "X-Tenant": ORG2})
    with pytest.raises(urllib.error.HTTPError) as info:
        urllib.request.urlopen(req, timeout=5)
    assert info.value.code == 403


def test_the_audit_holds_no_document_text_when_it_is_configured_to_omit(env):
    env.rag.cfg = dataclasses.replace(env.rag.cfg, audit_question="omit", audit_answer="omit")
    a = env.adapter()
    a.ingest(who(), [{"doc_id": "alfa.md", "text": ALFA, "acl": ["caso-a"]}])
    a.query(who(), Q_ALFA)
    dumped = repr(a.audit(who(user=USER3)))
    assert "5 de mayo" not in dumped and "Juzgado Tercero" not in dumped and "audiencia" not in dumped
