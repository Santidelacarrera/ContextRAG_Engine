"""End-to-end through the HTTP API with API keys: ingest, answers with sources, tenant isolation, update, deletion, and
the answer cache after an update or an ACL change. Synthetic documents only; the stack is the offline one (hashing
embeddings, in-memory stores), so this checks the contract, not model quality."""
import pytest
from fastapi.testclient import TestClient

from contextrag import factory
from contextrag.api.auth import KeyStore, new_entry
from contextrag.api.server import create_app

ALFA = "El expediente ALFA fija la audiencia preliminar el 5 de mayo de 2026 en el Juzgado Tercero."
ALFA_NEW = "El expediente ALFA fija la audiencia preliminar el 21 de octubre de 2026 en el Juzgado Cuarto."
BRAVO = "El expediente BRAVO fija la audiencia preliminar el 9 de junio de 2026 en el Juzgado Séptimo."
Q_ALFA = "¿Cuándo es la audiencia preliminar del expediente ALFA?"


@pytest.fixture
def stack():
    specs = [("root", "acme", "admin", ()), ("carl", "acme", "reader", ("caso-a",)),
             ("bob", "acme", "reader", ("caso-b",)), ("eve", "rival", "admin", ())]
    entries, plain = {}, {}
    for kid, tenant, role, groups in specs:
        key, digest, entry = new_entry(kid, tenant, role, groups)
        entries[digest], plain[kid] = entry, key
    rag = factory.offline()
    client = TestClient(create_app(rag, keystore=KeyStore(entries)))
    h = {who: {"X-API-Key": key} for who, key in plain.items()}
    yield client, h, rag
    rag.close()


def ingest(client, h, doc_id, text, acl, who="root"):
    r = client.post("/ingest", headers=h[who], json={"documents": [{"doc_id": doc_id, "text": text, "acl": acl}]})
    assert r.status_code == 200, r.text
    return r.json()


def ask(client, h, who, question=Q_ALFA):
    r = client.post("/query", headers=h[who], json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_ingest_then_query_answers_from_the_document_with_its_source(stack):
    client, h, _ = stack
    out = ingest(client, h, "alfa.md", ALFA, acl=["caso-a"])
    assert out["tenant"] == "acme" and out["documents"][0]["unchanged"] is False
    body = ask(client, h, "carl")
    assert body["status"] == "verified"
    assert "5 de mayo" in body["answer"]
    assert [s["doc_id"] for s in body["sources"]] == ["alfa.md"]
    assert body["sources"][0]["source"] == "alfa.md"


def test_a_key_of_another_tenant_gets_no_text_and_no_source_from_acme(stack):
    client, h, _ = stack
    ingest(client, h, "alfa.md", ALFA, acl=[])
    body = ask(client, h, "eve")
    assert body["sources"] == []
    assert "5 de mayo" not in body["answer"]
    assert "ALFA" not in body["answer"]


def test_a_reader_without_the_case_group_gets_no_text_from_the_restricted_document(stack):
    client, h, _ = stack
    ingest(client, h, "alfa.md", ALFA, acl=["caso-a"])
    body = ask(client, h, "bob")  # holds caso-b only
    assert body["sources"] == []
    assert "5 de mayo" not in body["answer"]


def test_an_update_replaces_the_answer_and_the_cached_old_answer_is_not_served(stack):
    client, h, _ = stack
    ingest(client, h, "alfa.md", ALFA, acl=[])
    assert "5 de mayo" in ask(client, h, "root")["answer"]
    assert ask(client, h, "root")["cached"] is True  # the old answer is in the cache
    ingest(client, h, "alfa.md", ALFA_NEW, acl=[])
    after = ask(client, h, "root")
    assert after["cached"] is False
    assert "21 de octubre" in after["answer"]
    assert "5 de mayo" not in after["answer"]


def test_a_deleted_document_stops_answering_and_leaves_the_listing(stack):
    client, h, _ = stack
    ingest(client, h, "alfa.md", ALFA, acl=[])
    assert ask(client, h, "root")["status"] == "verified"
    r = client.delete("/documents/alfa.md", headers=h["root"])
    assert r.status_code == 200 and r.json()["deleted"] == "alfa.md"
    after = ask(client, h, "root")
    assert after["status"] != "verified" and after["sources"] == []
    assert "5 de mayo" not in after["answer"]
    listed = client.get("/documents", headers=h["root"]).json()["documents"]
    assert all(d["doc_id"] != "alfa.md" for d in listed)


def test_an_acl_change_revokes_a_reader_who_could_read_the_answer_before(stack):
    client, h, _ = stack
    ingest(client, h, "alfa.md", ALFA, acl=["caso-a"])
    assert ask(client, h, "carl")["sources"]  # carl (caso-a) can read it, and the answer is now cached for him
    ingest(client, h, "alfa.md", ALFA, acl=["caso-b"])  # same text, different ACL: reindexed, not "unchanged"
    after = ask(client, h, "carl")
    assert after["sources"] == [] and "5 de mayo" not in after["answer"]
    assert "5 de mayo" in ask(client, h, "bob")["answer"]  # the new ACL grants caso-b


def test_the_answer_cites_the_document_it_came_from_when_two_documents_share_a_question_shape(stack):
    client, h, _ = stack
    ingest(client, h, "alfa.md", ALFA, acl=[])
    ingest(client, h, "bravo.md", BRAVO, acl=[])
    body = ask(client, h, "root", "¿Cuándo es la audiencia preliminar del expediente BRAVO?")
    assert body["status"] == "verified" and "9 de junio" in body["answer"]
    supported = [c for c in body["claims"] if c["supported"]]
    assert supported and all(c["evidence"]["doc_id"] == "bravo.md" for c in supported)
    # ``sources`` lists the passages given to the model, not only the cited ones: alfa.md may appear as context.
    assert body["sources"][0]["doc_id"] == "bravo.md"
