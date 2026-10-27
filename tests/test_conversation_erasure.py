"""Erasure reaches conversation memory: a turn built from a deleted document goes with it; a tenant's erasure removes its
conversations; other tenants and unrelated turns are untouched."""

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import factory
from contextrag.api.server import create_app, scoped_conversation_id
from contextrag.core.security import TenantCipher
from contextrag.memory.store import ConversationStore

KEY = TenantCipher(b"0123456789abcdef0123456789abcdef")
DOC = "El plazo de prescripción de la acción es de cuatro años contados desde la notificación."
OTHER = "El domicilio de la sociedad Mu es Valparaíso y su gerente general es Ana Díaz."


def rag_with_memory():
    r = factory.offline()
    r.memory = ConversationStore(":memory:", cipher=KEY)
    r.ingestor.ingest_text(DOC, tenant="t", doc_id="contrato.md", source="contrato.md")
    r.ingestor.ingest_text(OTHER, tenant="t", doc_id="mu.md", source="mu.md")
    return r


def turns(r, tenant, cid):
    return [t.content for t in r.memory.turns_between(tenant, cid, 0, 999)]


def test_deleting_a_document_removes_the_turn_built_from_it_and_keeps_the_rest_of_the_conversation():
    r = rag_with_memory()
    r.ask("¿Cuál es el plazo de prescripción?", tenant="t", conversation_id="c1")
    r.ask("¿Quién es el gerente de Mu?", tenant="t", conversation_id="c1")
    assert any("cuatro años" in x for x in turns(r, "t", "c1"))
    result = r.delete_document("t", "contrato.md")
    assert result["turns"] >= 1
    remaining = turns(r, "t", "c1")
    assert not any("cuatro años" in x for x in remaining)  # the excerpt is gone from memory
    assert any("Ana Díaz" in x for x in remaining)  # the unrelated answer is kept


def test_memories_come_from_user_statements_and_are_kept_by_design_until_the_conversation_is_erased():
    """The summariser promotes only user-role sentences, so a memory is text the user wrote, never a copy of a document
    the system produced. Removing the assistant turn that quoted a document does not remove such a memory. The user
    may still have pasted an excerpt into their own message; that is their content and is erased with the conversation."""
    from contextrag.memory.store import MemoryDraft

    r = rag_with_memory()
    r.memory.create_conversation("t", "c2")
    seq = r.memory.append_turn("t", "c2", "user", "Decidimos que el plazo de prescripción es de cuatro años.")
    r.memory.append_turn("t", "c2", "assistant", "Anotado: el plazo de prescripción es de cuatro años.", cites=["contrato.md"])
    job = r.memory.plan_compaction("t", "c2", threshold_tokens=0, keep_recent_turns=0)
    claim = r.memory.claim_job(job, lease_seconds=60)
    r.memory.complete_job(claim, [MemoryDraft("decision", "El plazo es de cuatro años.", "user_stated", 1.0, seq)])
    r.delete_document("t", "contrato.md")
    assert r.memory.memories("t", "c2")  # kept: its source is the user's own turn
    r.memory.delete_conversation("t", "c2")
    assert r.memory.memories("t", "c2") == []  # and gone with the conversation


def test_turns_written_before_provenance_was_recorded_are_reported_not_silently_assumed_clean():
    r = rag_with_memory()
    r.memory.create_conversation("t", "old")
    r.memory.append_turn("t", "old", "assistant", "El plazo es de cuatro años.")  # no cites: written before the change
    result = r.delete_document("t", "contrato.md")
    assert result["unlinked_assistant_turns"] >= 1
    assert "cuatro años" in " ".join(turns(r, "t", "old"))  # kept, and reported, so the operator can act


def test_another_tenants_conversations_are_never_touched():
    r = rag_with_memory()
    r.ingestor.ingest_text(DOC, tenant="rival", doc_id="contrato.md", source="contrato.md")
    r.ask("¿Cuál es el plazo de prescripción?", tenant="rival", conversation_id="c1")
    r.ask("¿Cuál es el plazo de prescripción?", tenant="t", conversation_id="c1")
    r.delete_document("t", "contrato.md")
    assert any("cuatro años" in x for x in turns(r, "rival", "c1"))


def test_tenant_erasure_removes_every_conversation_of_that_tenant_only():
    r = rag_with_memory()
    r.ask("¿Quién es el gerente de Mu?", tenant="t", conversation_id="c1")
    r.ask("¿Quién es el gerente de Mu?", tenant="rival", conversation_id="c1")
    out = r.erase_tenant("t")
    assert out["conversation_turns"] >= 2
    assert turns(r, "t", "c1") == []
    assert turns(r, "rival", "c1") != []


def test_the_api_deletion_reports_how_many_turns_were_removed():
    r = rag_with_memory()
    client = TestClient(create_app(r, keystore=None))
    client.post("/query", headers={"X-Tenant": "t"}, json={"question": "¿Cuál es el plazo de prescripción?",
                                                             "conversation_id": "c9"})
    body = client.delete("/documents/contrato.md", headers={"X-Tenant": "t"}).json()
    assert body["deleted"] == "contrato.md" and body["conversation_turns_removed"] >= 1
    scoped = scoped_conversation_id(None, "c9")
    assert not any("cuatro años" in x for x in turns(r, "t", scoped))
