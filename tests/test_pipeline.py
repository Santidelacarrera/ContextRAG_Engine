from contextrag import Status, factory
from contextrag.llm import ExtractiveLLM
from contextrag.verifier import parse_claims


def test_verified_answer_has_exact_evidence(rag):
    a = rag.ask("¿Cuánto tiempo se conservan los registros de auditoría?", tenant="acme")
    assert a.status is Status.VERIFIED and a.confidence >= 0.95
    ev = a.claims[0].evidence
    assert "7 años" in ev.text
    doc = rag.retriever.store.parent(a.sources[0].parent.id, "acme")
    assert doc.text  # evidence coordinates are absolute document offsets
    assert ev.start >= doc.start and ev.end <= doc.end


def test_unrelated_question_falls_back(rag):
    a = rag.ask("¿Quién ganó el mundial de 1998?", tenant="acme")
    assert a.status is Status.FALLBACK and a.text


class HallucinatingLLM:
    def generate(self, question, contexts, feedback=None):
        return "Los registros de auditoría se conservan durante 10 años. [1]"


def test_hallucinated_number_is_rejected():
    r = factory.offline(llm=HallucinatingLLM())
    r.ingestor.ingest_text("Los registros de auditoría se conservan durante 7 años.", tenant="t", doc_id="d")
    a = r.ask("¿Cuánto se conservan los registros de auditoría?", tenant="t")
    assert a.status is Status.FALLBACK and a.attempts == 2
    assert "10" not in a.text


class UncitedLLM:
    def generate(self, question, contexts, feedback=None):
        return "Los registros de auditoría se conservan durante 7 años."


def test_uncited_claim_is_rejected():
    r = factory.offline(llm=UncitedLLM())
    r.ingestor.ingest_text("Los registros de auditoría se conservan durante 7 años.", tenant="t", doc_id="d")
    assert r.ask("registros de auditoría", tenant="t").status is Status.FALLBACK


class SelfCorrectingLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, question, contexts, feedback=None):
        self.calls += 1
        if feedback is None:
            return "Los registros de auditoría se conservan durante 10 años. [1]"
        return ExtractiveLLM().generate(question, contexts)


def test_self_correction_loop_recovers():
    llm = SelfCorrectingLLM()
    r = factory.offline(llm=llm)
    r.ingestor.ingest_text("Los registros de auditoría se conservan durante 7 años.", tenant="t", doc_id="d")
    a = r.ask("¿Cuánto se conservan los registros de auditoría?", tenant="t")
    assert a.status is Status.VERIFIED and a.attempts == 2 and llm.calls == 2


def test_tenant_and_acl_isolation(rag):
    assert rag.ask("registros de auditoría", tenant="other").status is Status.FALLBACK
    rag.ingestor.ingest_text("El código secreto del proyecto es ORIÓN.", tenant="acme", doc_id="s", acl=("legal",))
    q = "¿Cuál es el código secreto del proyecto?"
    assert rag.ask(q, tenant="acme").status is Status.FALLBACK
    assert rag.ask(q, tenant="acme", groups=frozenset({"legal"})).status is Status.VERIFIED


def test_parse_claims_markers():
    assert parse_claims("A es B. [1] C es D [2][3].") == [("A es B.", [1]), ("C es D.", [2, 3])]


def test_audit_written(rag):
    rag.ask("¿Cómo se cifran los datos en reposo?", tenant="acme", user="u1")
    rec = rag.audit.list("acme")[0]
    assert rec["status"] == "verified" and rec["user"] == "u1" and rec["claims"][0]["evidence"]["text"]
