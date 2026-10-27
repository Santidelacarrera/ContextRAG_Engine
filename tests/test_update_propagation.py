"""An updated document must stop answering with its old content everywhere: index, BM25, and cached answers."""
from contextrag.factory import offline
from contextrag.core.models import Status

Q = "¿cuánto tiempo se conservan los registros?"


def test_an_updated_document_no_longer_answers_with_its_old_text_even_from_cache():
    rag = offline()
    try:
        rag.ingestor.ingest_text("Los registros se conservan 7 años.", tenant="t", doc_id="d.md", source="d.md")
        first = rag.ask(Q, tenant="t")
        assert first.status is Status.VERIFIED and "7 años" in first.text
        again = rag.ask(Q, tenant="t")
        assert again.cached is True  # the old answer is in the cache before the update

        rag.ingestor.ingest_text("Los registros se conservan 9 años.", tenant="t", doc_id="d.md", source="d.md")

        after = rag.ask(Q, tenant="t")
        assert after.cached is False  # the tenant epoch moved: the old cache entry is unreachable
        assert "7 años" not in after.text
        assert "9 años" in after.text
    finally:
        rag.close()


def test_a_deleted_document_is_absent_from_retrieval_and_from_the_cache():
    rag = offline()
    try:
        rag.ingestor.ingest_text("Los registros se conservan 7 años.", tenant="t", doc_id="d.md", source="d.md")
        assert rag.ask(Q, tenant="t").status is Status.VERIFIED
        rag.delete_document("t", "d.md")
        after = rag.ask(Q, tenant="t")
        assert after.status is not Status.VERIFIED
        assert "7 años" not in (after.text or "")
    finally:
        rag.close()
