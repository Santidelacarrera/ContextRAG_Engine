"""Case separation inside one tenant. A case is an ACL group; a user is bound to the cases (groups) of their key.

Every test here is negative: an authorised user of one case tries to reach another case and must get a safe answer
without any trace of the other case's content. Data is synthetic.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import factory
from contextrag.api.auth import KeyStore, new_entry
from contextrag.api.server import create_app
from contextrag.core.security import TenantCipher
from contextrag.memory.store import ConversationStore

A_TEXT = "El expediente ALFA fija la audiencia preliminar el 5 de mayo de 2026 en el Juzgado Tercero."
B_TEXT = "El expediente BRAVO fija la audiencia preliminar el 9 de junio de 2026 en el Juzgado Séptimo."
SECRET_B = "BRAVO"
# Content that exists only in case B's document. The word BRAVO itself is excluded on purpose: the response echoes the term
# the caller typed in their own question, which is not information from the other case.
B_ONLY = ("9 de junio", "Séptimo", "2026 en el Juzgado")


def no_case_b_content(text):
    assert not any(fragment in text for fragment in B_ONLY)


def build(extra_rag=None):
    specs = [("root", "acme", "admin", ()), ("alice", "acme", "admin", ("caso-a",)),
             ("bob", "acme", "admin", ("caso-b",)), ("carl", "acme", "reader", ("caso-a",)),
             ("zed", "acme", "reader", ("caso-a", "caso-b")), ("eve", "rival", "admin", ())]
    entries, plain = {}, {}
    for kid, tenant, role, groups in specs:
        key, digest, entry = new_entry(kid, tenant, role, groups)
        entries[digest], plain[kid] = entry, key
    rag = extra_rag or factory.offline()
    client = TestClient(create_app(rag, keystore=KeyStore(entries)))
    h = lambda who: {"X-API-Key": plain[who]}
    for doc_id, text, acl in (("a.md", A_TEXT, ["caso-a"]), ("b.md", B_TEXT, ["caso-b"])):
        r = client.post("/ingest", headers=h("root"), json={"documents": [{"doc_id": doc_id, "text": text, "acl": acl}]})
        assert r.status_code == 200, r.text
    return client, h, rag


@pytest.fixture
def world():
    return build()


Q_B = {"question": "¿Cuándo es la audiencia preliminar del expediente BRAVO?"}
Q_A = {"question": "¿Cuándo es la audiencia preliminar del expediente ALFA?"}


# ---- reads -------------------------------------------------------------------------------------

def test_a_user_of_one_case_cannot_read_another_case_through_query(world):
    client, h, _ = world
    r = client.post("/query", headers=h("carl"), json=Q_B)
    assert r.status_code == 200 and r.json()["status"] == "fallback"
    no_case_b_content(r.text)
    resolution = r.json()["meta"]["entity_resolution"]
    assert resolution == [{"term": "BRAVO", "outcome": "unknown", "canonical": []}]  # nothing about case B is revealed
    assert client.post("/query", headers=h("carl"), json=Q_A).json()["status"] == "verified"


def test_a_request_cannot_widen_the_callers_groups(world):
    client, h, _ = world
    r = client.post("/query", headers=h("carl"), json={**Q_B, "groups": ["caso-b"]})
    assert r.status_code == 403 and SECRET_B not in r.text


def test_retrieve_returns_only_the_callers_cases(world):
    client, h, _ = world
    r = client.post("/api/v1/retrieve", headers=h("carl"),
                    json={"tenant_id": "acme", "query": "audiencia preliminar expediente", "top_k": 10})
    assert r.status_code in (200, 503)  # 503 only if the audit database is required and absent
    if r.status_code == 200:
        assert SECRET_B not in r.text and "9 de junio" not in r.text


def test_cached_answers_do_not_cross_cases(world):
    client, h, _ = world
    both = client.post("/query", headers=h("zed"), json=Q_B)  # zed sees both cases and warms the cache
    assert both.json()["status"] == "verified"
    carl = client.post("/query", headers=h("carl"), json=Q_B)
    assert carl.json()["status"] == "fallback"
    no_case_b_content(carl.text)


def test_another_tenant_sees_nothing(world):
    client, h, _ = world
    r = client.post("/query", headers=h("eve"), json=Q_A)
    assert r.json()["status"] == "fallback" and "ALFA" not in r.json()["answer"] and "5 de mayo" not in r.text


# ---- writes ------------------------------------------------------------------------------------

def ingest(client, headers, doc_id, text, acl):
    return client.post("/ingest", headers=headers, json={"documents": [{"doc_id": doc_id, "text": text, "acl": acl}]})


def test_a_case_admin_cannot_create_documents_for_other_cases_or_for_the_whole_tenant(world):
    client, h, _ = world
    assert ingest(client, h("alice"), "n.md", "texto", ["caso-b"]).status_code == 403
    assert ingest(client, h("alice"), "n.md", "texto", ["caso-a", "caso-b"]).status_code == 403
    assert ingest(client, h("alice"), "n.md", "texto", []).status_code == 403  # tenant-wide needs an unrestricted admin
    assert ingest(client, h("alice"), "n.md", "texto de prueba suficiente", ["caso-a"]).status_code == 200


def test_a_case_admin_cannot_overwrite_a_document_of_another_case(world):
    client, h, _ = world
    r = ingest(client, h("alice"), "b.md", "El expediente BRAVO fija la audiencia el 1 de enero de 2030.", ["caso-a"])
    assert r.status_code in (403, 404)
    ok = client.post("/query", headers=h("bob"), json=Q_B).json()
    assert ok["status"] == "verified" and "9 de junio" in ok["answer"] and "2030" not in ok["answer"]


def test_a_case_admin_cannot_delete_another_cases_document_and_gets_no_existence_oracle(world):
    client, h, _ = world
    forbidden = client.delete("/documents/b.md", headers=h("alice"))
    missing = client.delete("/documents/does-not-exist.md", headers=h("alice"))
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()  # identical: forbidden and absent cannot be told apart
    assert client.post("/query", headers=h("bob"), json=Q_B).json()["status"] == "verified"
    assert client.delete("/documents/a.md", headers=h("alice")).status_code == 200


def test_listing_shows_only_the_callers_cases(world):
    client, h, _ = world
    docs = client.get("/documents", headers=h("alice")).json()["documents"]
    assert [d["doc_id"] for d in docs] == ["a.md"]
    assert {d["doc_id"] for d in client.get("/documents", headers=h("root")).json()["documents"]} == {"a.md", "b.md"}


def test_only_unrestricted_admins_can_erase_a_tenant(world):
    client, h, _ = world
    r = client.delete("/tenant", headers={**h("alice"), "X-Confirm-Tenant": "acme"})
    assert r.status_code == 403
    assert client.post("/query", headers=h("bob"), json=Q_B).json()["status"] == "verified"


def test_the_audit_log_of_a_case_admin_contains_only_their_own_queries(world):
    client, h, _ = world
    client.post("/query", headers=h("bob"), json=Q_B)
    client.post("/query", headers=h("alice"), json=Q_A)
    mine = client.get("/audit", headers=h("alice")).json()
    assert mine and all(rec["user"] == "alice" for rec in mine)
    assert SECRET_B not in str(mine)
    everything = client.get("/audit", headers=h("root")).json()
    assert {r["user"] for r in everything} >= {"alice", "bob"}


# ---- conversations (identifiers must not be a capability) ----------------------------------------

def test_a_conversation_id_is_not_a_capability_within_a_tenant():
    rag = factory.offline()
    rag.memory = ConversationStore(":memory:", cipher=TenantCipher(b"0123456789abcdef0123456789abcdef"))
    client, h, _ = build(rag)
    try:
        client.post("/query", headers=h("zed"), json={**Q_B, "conversation_id": "shared-id"})
        # carl guesses the id: he must get his own empty conversation, not zed's history
        r = client.post("/query", headers=h("carl"), json={"question": "Resume lo anterior", "conversation_id": "shared-id"})
        assert r.status_code == 200 and SECRET_B not in r.text and "9 de junio" not in r.text
        assert client.delete("/conversations/shared-id", headers=h("bob")).status_code == 200  # erase is admin-only
        owner_turns = rag.memory.history_window("acme", _scoped(rag, "zed", "shared-id"), 10_000)
        assert owner_turns, "carl's delete must not have removed zed's conversation"
    finally:
        rag.close()


def _scoped(rag, user, cid):
    from contextrag.api.server import scoped_conversation_id

    return scoped_conversation_id(user, cid)


# ---- credentials and mode ---------------------------------------------------------------------------

def test_invalid_or_insufficient_credentials_fail_safely(world):
    client, h, _ = world
    for headers in ({}, {"X-API-Key": "crag_wrong"}, {"X-API-Key": ""}):
        r = client.get("/documents", headers=headers)
        assert r.status_code == 401 and "acme" not in r.text and "caso" not in r.text
    assert client.post("/ingest", headers=h("carl"), json={"documents": [{"doc_id": "x.md", "text": "texto"}]}).status_code == 403
    assert client.delete("/documents/a.md", headers=h("carl")).status_code == 403


def test_strict_mode_refuses_to_start_without_a_key_store(monkeypatch):
    monkeypatch.setenv("CONTEXTRAG_REQUIRE_AUTH", "true")
    monkeypatch.delenv("CONTEXTRAG_KEYS_FILE", raising=False)
    with pytest.raises(RuntimeError, match="CONTEXTRAG_KEYS_FILE"):
        create_app(factory.offline(), keystore=None)
    ks = KeyStore({})
    create_app(factory.offline(), keystore=ks)  # an explicit key store satisfies the requirement
