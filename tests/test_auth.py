import json
import os
import stat

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import factory
from contextrag.__main__ import main
from contextrag.api.auth import KeyStore, hash_key, new_entry
from contextrag.api.ratelimit import RateLimiter
from contextrag.api.server import create_app
from conftest import POLICY

Q = {"question": "¿Cuánto tiempo se conservan los registros de auditoría?"}


def keys(*specs):
    """specs: (id, tenant, role, groups) -> (KeyStore, {id: plaintext key})"""
    entries, plain = {}, {}
    for kid, tenant, role, groups in specs:
        key, digest, entry = new_entry(kid, tenant, role, groups)
        entries[digest], plain[kid] = entry, key
    return KeyStore(entries), plain


@pytest.fixture
def secured():
    ks, k = keys(("ana", "acme", "admin", ()), ("bob", "acme", "reader", ()), ("lia", "acme", "reader", ("legal",)),
                 ("eve", "rival", "admin", ()), ("ops", "", "ops", ()))
    c = TestClient(create_app(factory.offline(), keystore=ks))
    h = lambda who, **extra: {"X-API-Key": k[who], **extra}
    r = c.post("/ingest", json={"documents": [{"doc_id": "policy.md", "text": POLICY},
                                              {"doc_id": "s.txt", "text": "El código secreto es ORIÓN.", "acl": ["legal"]}]},
               headers=h("ana"))
    assert r.status_code == 200
    return c, h, k


def test_requests_without_or_with_wrong_key_are_rejected(secured):
    c, h, _ = secured
    assert c.post("/query", json=Q).status_code == 401
    assert c.post("/query", json=Q, headers={"X-API-Key": "crag_nope"}).status_code == 401
    assert c.post("/query", json=Q, headers={"X-Tenant": "acme"}).status_code == 401  # a tenant claim is not credentials
    assert c.get("/healthz").status_code == 200  # probes stay open


def test_tenant_comes_from_the_key_not_from_the_caller(secured):
    c, h, _ = secured
    assert c.post("/query", json=Q, headers=h("bob")).json()["status"] == "verified"
    assert c.post("/query", json=Q, headers=h("bob", **{"X-Tenant": "acme"})).status_code == 200
    assert c.post("/query", json=Q, headers=h("bob", **{"X-Tenant": "rival"})).status_code == 403
    rival = c.post("/query", json=Q, headers=h("eve")).json()  # eve's key only reaches tenant "rival"
    assert rival["status"] == "fallback" and rival["sources"] == []


def test_roles_limit_what_a_key_can_do(secured):
    c, h, _ = secured
    doc = {"documents": [{"doc_id": "x", "text": "y"}]}
    assert c.post("/ingest", json=doc, headers=h("bob")).status_code == 403  # reader cannot write
    assert c.delete("/documents/policy.md", headers=h("bob")).status_code == 403
    assert c.get("/audit", headers=h("bob")).status_code == 403
    assert c.get("/metrics", headers=h("bob")).status_code == 403
    assert c.post("/query", json=Q, headers=h("ops")).status_code == 403  # ops cannot read tenant data
    assert c.get("/metrics", headers=h("ops")).status_code == 200
    assert c.post("/ingest", json=doc, headers=h("ana")).status_code == 200
    assert c.get("/audit", headers=h("ana")).status_code == 200


def test_acl_groups_are_bound_to_the_key_and_cannot_be_self_granted(secured):
    c, h, _ = secured
    secret = {"question": "¿Cuál es el código secreto?"}
    assert c.post("/query", json=secret, headers=h("bob")).json()["status"] == "fallback"
    # bob claims the group in the body: refused instead of silently honoured
    assert c.post("/query", json={**secret, "groups": ["legal"]}, headers=h("bob")).status_code == 403
    assert c.post("/query", json=secret, headers=h("lia")).json()["status"] == "verified"  # lia's key carries 'legal'
    assert c.post("/query", json={**secret, "groups": ["legal"]}, headers=h("lia")).json()["status"] == "verified"
    assert c.post("/query", json={**secret, "groups": ["hr"]}, headers=h("lia")).status_code == 403


def test_groups_can_be_narrowed(secured):
    c, h, _ = secured
    secret = {"question": "¿Cuál es el código secreto?"}
    # an empty list means "all of the key's groups"; naming a different (granted) subset narrows to it
    assert c.post("/query", json=secret, headers=h("lia")).json()["status"] == "verified"


def test_audit_records_the_key_id_not_a_spoofable_header(secured):
    c, h, _ = secured
    c.post("/query", json=Q, headers=h("bob", **{"X-User": "mallory"}))
    rows = c.get("/audit", headers=h("ana")).json()
    assert rows[0]["user"] == "bob"
    assert c.get("/audit", headers=h("eve")).json() == []  # other tenant's admin sees nothing of acme


def test_keystore_stores_only_digests_and_validates_entries():
    key, digest, entry = new_entry("a", "acme", "reader")
    assert digest == hash_key(key) and key not in json.dumps({digest: entry})
    assert KeyStore({digest: entry}).authenticate(key).tenant == "acme"
    assert KeyStore({digest: entry}).authenticate(key + "x") is None
    with pytest.raises(ValueError):
        KeyStore({digest: {"id": "a", "role": "reader"}})  # reader without a tenant
    with pytest.raises(ValueError):
        KeyStore({digest: {"id": "a", "tenant": "t", "role": "root"}})
    with pytest.raises(ValueError):
        new_entry("a", "t", "root")


def test_keys_file_loaded_from_env_and_keygen_cli(tmp_path, monkeypatch, capsys):
    f = tmp_path / "keys.json"
    assert main(["keygen", "--id", "ana", "--tenant", "acme", "--role", "admin", "--add-to", str(f)]) == 0
    admin_key = capsys.readouterr().out.strip()
    assert main(["keygen", "--id", "lia", "--tenant", "acme", "--groups", "legal,hr", "--add-to", str(f)]) == 0
    reader_key = capsys.readouterr().out.strip()
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600
    stored = f.read_text()
    assert admin_key not in stored and reader_key not in stored and hash_key(admin_key) in stored
    monkeypatch.setenv("CONTEXTRAG_KEYS_FILE", str(f))
    c = TestClient(create_app(factory.offline()))
    assert c.post("/ingest", json={"documents": [{"doc_id": "p.md", "text": POLICY}]}, headers={"X-API-Key": admin_key}).status_code == 200
    assert c.post("/query", json=Q, headers={"X-API-Key": reader_key}).json()["status"] == "verified"
    with pytest.raises(ValueError):
        main(["keygen", "--id", "x", "--role", "reader"])  # tenant required


# ---- rate limiting -----------------------------------------------------------------------

def test_token_bucket_refills_over_time_and_is_per_key():
    now = [0.0]
    rl = RateLimiter(per_minute=60, burst=2, clock=lambda: now[0])  # 1 token/second
    assert rl.acquire("a") == 0 and rl.acquire("a") == 0
    wait = rl.acquire("a")
    assert 0.9 < wait <= 1.0
    assert rl.acquire("b") == 0  # separate bucket
    now[0] += 1.0
    assert rl.acquire("a") == 0 and rl.acquire("a") > 0
    now[0] += 100
    assert [rl.acquire("a") for _ in range(3)][:2] == [0, 0] and rl.acquire("a") > 0  # capped at burst


def test_api_returns_429_with_retry_after_and_limits_per_tenant():
    c = TestClient(create_app(factory.offline(), rate_limit_per_min=3))
    h = {"X-Tenant": "acme"}
    codes = [c.post("/query", json={"question": "x"}, headers=h).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    r = c.post("/query", json={"question": "x"}, headers=h)
    assert int(r.headers["Retry-After"]) >= 1
    assert c.post("/query", json={"question": "x"}, headers={"X-Tenant": "other"}).status_code == 200
    assert c.get("/healthz").status_code == 200  # never limited


def test_only_admins_can_list_documents(secured):
    c, h, _ = secured
    assert c.get("/documents", headers=h("bob")).status_code == 403
    assert c.get("/documents", headers=h("ops")).status_code == 403
    docs = c.get("/documents", headers=h("ana")).json()["documents"]
    assert [d["doc_id"] for d in docs] == ["policy.md", "s.txt"]
    assert c.get("/documents", headers=h("eve")).json()["documents"] == []
