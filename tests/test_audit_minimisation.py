"""Audit minimisation: with ``audit_question``/``audit_answer`` set to ``omit`` the audit log keeps the length and the
status of a query, never its text. The default keeps the text, as before."""
import pytest

from contextrag.audit import record_of
from contextrag.core.config import Settings
from contextrag.core.models import Answer, Status
from contextrag.factory import offline

SECRET_QUESTION = "¿Cuál es la cláusula confidencial de la fusión Zeta-7?"
SECRET_ANSWER = "La cláusula confidencial fija una penalización de 1234 UF."


def _answer():
    return Answer(question=SECRET_QUESTION, status=Status.VERIFIED, text=SECRET_ANSWER, confidence=0.9, attempts=1)


def test_the_default_keeps_the_text_as_before():
    rec = record_of("acme", None, _answer())
    assert rec["question"] == SECRET_QUESTION and rec["answer"] == SECRET_ANSWER


def test_omit_keeps_only_the_length_and_the_status():
    rec = record_of("acme", None, _answer(), question_mode="omit", answer_mode="omit")
    assert SECRET_QUESTION not in str(rec) and SECRET_ANSWER not in str(rec)
    assert rec["question"] == f"[omitted: {len(SECRET_QUESTION)} chars]"
    assert rec["answer"] == f"[omitted: {len(SECRET_ANSWER)} chars]"
    assert rec["status"] == "verified"


def test_the_pipeline_writes_no_query_text_when_configured_to_omit():
    rag = offline(settings=Settings(audit_question="omit", audit_answer="omit"))
    try:
        rag.ingestor.ingest_text("La cláusula confidencial fija una penalización de 1234 UF.", tenant="acme", doc_id="d.md",
                                 source="d.md")
        rag.ask(SECRET_QUESTION, tenant="acme")
        rows = rag.audit.list("acme")
        assert rows, "the query must still be audited"
        dumped = repr(rows)
        assert "Zeta-7" not in dumped and "1234" not in dumped
        assert rows[0]["question"].startswith("[omitted:")
    finally:
        rag.close()


def test_the_pipeline_keeps_text_by_default():
    rag = offline()
    try:
        rag.ingestor.ingest_text("La cláusula confidencial fija una penalización de 1234 UF.", tenant="acme", doc_id="d.md",
                                 source="d.md")
        rag.ask(SECRET_QUESTION, tenant="acme")
        assert rag.audit.list("acme")[0]["question"] == SECRET_QUESTION
    finally:
        rag.close()


@pytest.mark.parametrize("value", ["digest", "", "none", "FULL"])
def test_an_unknown_mode_is_refused_at_configuration(value):
    with pytest.raises(ValueError):
        Settings(audit_question=value)
    with pytest.raises(ValueError):
        Settings(audit_answer=value)


def test_the_environment_variables_reach_settings(monkeypatch):
    monkeypatch.setenv("CONTEXTRAG_AUDIT_QUESTION", "omit")
    monkeypatch.setenv("CONTEXTRAG_AUDIT_ANSWER", "omit")
    cfg = Settings.from_env()
    assert (cfg.audit_question, cfg.audit_answer) == ("omit", "omit")


def test_omit_also_drops_the_claim_sentences_that_repeat_the_answer():
    # A verified answer's claims are its sentences. Keeping them under "omit" would store the answer text by another name.
    secret_claim = "La cláusula confidencial fija una penalización de 1234 UF."
    rag = offline(settings=Settings(audit_question="omit", audit_answer="omit"))
    try:
        rag.ingestor.ingest_text(secret_claim, tenant="acme", doc_id="d.md", source="d.md")
        a = rag.ask("¿Cuál es la penalización de la cláusula confidencial?", tenant="acme")
        assert a.claims, "the scenario must produce at least one claim"
        row = rag.audit.list("acme")[0]
        assert "1234" not in repr(row) and "penalización" not in repr(row)
        assert all(c["claim"].startswith("[omitted:") for c in row["claims"])
        assert all("supported" in c and "citations" in c for c in row["claims"])  # verdicts and citations are kept
    finally:
        rag.close()
