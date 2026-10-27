"""Deleting a document removes everything derived from it that the system still controls. Regression for the audit gap:
the audit used to keep the verbatim evidence excerpt after the document was deleted."""
from contextrag import Status, factory
from contextrag.audit import InMemoryAudit

FACT = "El plazo de prescripción de la acción es de cuatro años contados desde la notificación."
Q = "¿Cuál es el plazo de prescripción de la acción?"


def indexed():
    from contextrag.retrieval.vector_store import InMemoryVectorStore  # noqa: F401  (documents the store under test)

    r = factory.offline()
    r.audit = InMemoryAudit()
    r.ingestor.ingest_text(FACT, tenant="t", doc_id="contrato.md", source="contrato.md")
    return r


def test_after_deletion_the_audit_holds_no_excerpt_but_still_proves_what_was_cited():
    r = indexed()
    a = r.ask(Q, tenant="t")
    assert a.status is Status.VERIFIED
    r.ingestor.delete(tenant="t", doc_id="contrato.md")
    record = r.audit.list("t")[0]
    ev = record["claims"][0]["evidence"]
    assert "text" not in ev  # the excerpt itself is not kept in the evidence record
    assert ev["doc_id"] == "contrato.md" and ev["text_sha256"] and ev["text_chars"] > 0  # the reference survives
    # the answer and claim sentences are generated text, kept as the record of what was said (see the last test)


def test_after_deletion_the_document_is_neither_searchable_nor_quotable_from_cache():
    r = indexed()
    assert r.ask(Q, tenant="t").status is Status.VERIFIED
    assert r.ask(Q, tenant="t").cached is True
    r.ingestor.delete(tenant="t", doc_id="contrato.md")
    gone = r.ask(Q, tenant="t")
    assert gone.cached is False and gone.status is Status.FALLBACK and "cuatro años" not in gone.text
    assert r.index.store.dense(r.embedder.embed([FACT])[0], "t", frozenset(), 5) == []
    assert r.index.bm25.search("prescripción", "t", frozenset(), 5) == []


def test_the_audit_digest_matches_the_original_excerpt_when_the_document_is_still_available():
    import hashlib

    r = indexed()
    a = r.ask(Q, tenant="t")
    rec = r.audit.list("t")[0]
    original = a.claims[0].evidence.text
    assert rec["claims"][0]["evidence"]["text_sha256"] == hashlib.sha256(original.encode()).hexdigest()


def test_the_known_remaining_copy_is_the_generated_answer_text_and_is_documented():
    """The answer itself can repeat document wording. Removing it from the audit would remove the record of what was said,
    which accountability needs, so it is kept and called out in the documentation rather than hidden."""
    r = indexed()
    r.ask(Q, tenant="t")
    r.ingestor.delete(tenant="t", doc_id="contrato.md")
    assert "cuatro años" in r.audit.list("t")[0]["answer"]
