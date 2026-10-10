from contextrag import Status, factory
from contextrag.core.cache import InMemoryCache
from contextrag.core.models import Answer
from contextrag.generation.llm import ExtractiveLLM
from contextrag.verification.nli_verifier import fallback_message

Q = "¿Cuánto tiempo se conservan los registros de auditoría?"


def make(llm, text="Los registros de auditoría se conservan durante 7 años.", tenant="t"):
    r = factory.offline(llm=llm)
    r.ingestor.ingest_text(text, tenant=tenant, doc_id="d.md")
    return r


def test_verified_answer_has_exact_evidence(rag):
    a = rag.ask(Q, tenant="acme")
    assert a.status is Status.VERIFIED and a.confidence >= 0.95
    ev = a.claims[0].evidence
    assert "7 años" in ev.text
    src = next(c.parent for c in a.sources if c.parent.doc_id == ev.doc_id)
    assert src.start <= ev.start and ev.end <= src.end


def test_unrelated_question_falls_back_with_spec_message(rag):
    a = rag.ask("¿Quién ganó el mundial de 1998?", tenant="acme")
    assert a.status is Status.FALLBACK and a.text == fallback_message()


class Lying:
    def generate(self, question, contexts, feedback=None):
        return "Los registros de auditoría se conservan durante 10 años. [1]"


def test_hallucinated_number_is_rejected_after_retries():
    a = make(Lying()).ask(Q, tenant="t")
    assert a.status is Status.FALLBACK and a.attempts == 2 and "10" not in a.text
    assert a.confidence == 0.0 and a.claims and not a.claims[0].supported


def test_uncited_claim_is_rejected():
    class Uncited:
        def generate(self, *a, **k):
            return "Los registros de auditoría se conservan durante 7 años."

    assert make(Uncited()).ask(Q, tenant="t").status is Status.FALLBACK


def test_self_correction_loop_recovers_and_passes_feedback():
    class Fixer:
        feedbacks: list = []

        def generate(self, question, contexts, feedback=None):
            self.feedbacks.append(feedback)
            return Lying().generate(0, 0) if feedback is None else ExtractiveLLM().generate(question, contexts)

    llm = Fixer()
    a = make(llm).ask(Q, tenant="t")
    assert a.status is Status.VERIFIED and a.attempts == 2
    assert llm.feedbacks[0] is None and "10 años" in llm.feedbacks[1]


def test_model_declining_triggers_fallback_without_retry():
    class Decline:
        calls = 0

        def generate(self, *a, **k):
            Decline.calls += 1
            return "NO_ANSWER"

    a = make(Decline()).ask(Q, tenant="t")
    assert a.status is Status.FALLBACK and Decline.calls == 1


def test_tenant_and_acl_isolation(rag):
    assert rag.ask("registros de auditoría", tenant="other").status is Status.FALLBACK
    rag.ingestor.ingest_text("El código secreto del proyecto es ORIÓN.", tenant="acme", doc_id="s.txt", acl=("legal",))
    q = "¿Cuál es el código secreto del proyecto?"
    assert rag.ask(q, tenant="acme").status is Status.FALLBACK
    assert rag.ask(q, tenant="acme", groups=frozenset({"legal"})).status is Status.VERIFIED


def test_audit_record_contains_evidence_and_sources(rag):
    rag.ask("¿Cómo se cifran los datos en reposo?", tenant="acme", user="u1")
    rec = rag.audit.list("acme")[0]
    assert rec["status"] == "verified" and rec["user"] == "u1"
    assert rec["claims"][0]["evidence"]["text"] and rec["sources"][0]["doc_id"] == "policy.md"
    assert rag.audit.list("other") == []


def test_answer_roundtrips_through_json(rag):
    a = rag.ask(Q, tenant="acme")
    import json

    b = Answer.from_dict(json.loads(json.dumps(a.to_dict())))
    assert b == a


def test_query_cache_hits_and_is_invalidated_by_ingest(rag):
    first = rag.ask(Q, tenant="acme")
    again = rag.ask(Q, tenant="acme")
    assert not first.cached and again.cached and again.text == first.text
    rag.ingestor.ingest_text("Los registros de auditoría se conservan durante 9 años.", tenant="acme", doc_id="policy.md")
    after = rag.ask(Q, tenant="acme")
    assert not after.cached and "9 años" in after.text


def test_cache_is_scoped_by_tenant_and_groups(rag):
    rag.ask(Q, tenant="acme")
    assert rag.ask(Q, tenant="other").status is Status.FALLBACK  # a cached acme answer must not leak
    assert not rag.ask(Q, tenant="acme", groups=frozenset({"x"})).cached


def test_fallbacks_are_not_cached():
    r = make(Lying())
    r.ask(Q, tenant="t")
    assert not r.ask(Q, tenant="t").cached


def test_pipeline_without_cache_or_audit():
    r = factory.offline()
    r.cache = r.audit = None
    r.ingestor.ingest_text("El cielo es azul.", tenant="t", doc_id="x.txt")
    assert r.ask("color del cielo", tenant="t").status is Status.VERIFIED
    assert r.health() == {"index": True}


def test_markdown_and_code_documents_are_answerable():
    r = factory.offline()
    r.ingestor.ingest_text("# API\n\nEl timeout por defecto es de 30 segundos.", tenant="t", doc_id="api.md")
    r.ingestor.ingest_text("def retry_limit():\n    # El límite de reintentos es 5.\n    return 5\n", tenant="t", doc_id="cfg.py")
    a = r.ask("¿Cuál es el timeout por defecto?", tenant="t")
    assert a.status is Status.VERIFIED and a.sources[0].parent.section == "API"
