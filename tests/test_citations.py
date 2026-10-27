"""Traceability of answers: citations are checked against the original documents, apart from retrieval and apart from truth.

Three things are kept separate on purpose. Retrieval quality is measured by the retrieval benchmark. Whether a claim is
*true* is not something the system can know. What is checked here is that an answer can be traced to passages that were
retrieved, that its quotations are exact, and that nothing in it is invented.
"""
import copy
import json
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from conftest import make_pdf
from contextrag import Status, factory
from contextrag.api.server import create_app
from contextrag.core.config import Settings
from contextrag.core.ingestion import load_bytes
from contextrag.eval import entity_bench as E
from contextrag.eval import retrieval_bench as B
from contextrag.eval.citations import citation_problems


# ---- every verified answer in both benchmarks is traceable ------------------------------------------------------

def all_answers(dataset_loader, ask_kwargs=lambda q: {}):
    data = dataset_loader()
    rag = factory.offline()
    docs = {d["doc_id"]: d["text"] for d in data["documents"]}
    for d in data["documents"]:
        rag.ingestor.ingest_text(d["text"], tenant="t", doc_id=d["doc_id"], source=d["doc_id"], acl=tuple(d.get("acl", ())))
    return [(q, rag.ask(q["question"], tenant="t", **ask_kwargs(q))) for q in data["queries"]], docs


def test_every_verified_answer_of_the_retrieval_benchmark_has_valid_citations():
    results, docs = all_answers(lambda: B.load_dataset()[0])
    verified = [(q, a) for q, a in results if a.status is Status.VERIFIED]
    assert len(verified) >= 30
    for q, a in verified:
        assert citation_problems(a, docs) == [], (q["id"], citation_problems(a, docs))
        assert a.claims and all(c.supported and c.evidence for c in a.claims)


def test_every_verified_answer_of_the_entity_benchmark_has_valid_citations():
    results, docs = all_answers(lambda: json.loads(E.DATA_PATH.read_text(encoding="utf-8")),
                                lambda q: {"groups": frozenset(q["groups"])})
    verified = [(q, a) for q, a in results if a.status is Status.VERIFIED]
    assert len(verified) >= 15
    for q, a in verified:
        assert citation_problems(a, docs) == [], q["id"]


# ---- the auditor itself catches what it should --------------------------------------------------------------------

def tampered(answer, **changes):
    """A copy of ``answer`` whose first claim's (immutable) evidence has the given fields replaced."""
    bad = copy.deepcopy(answer)
    bad.claims[0].evidence = replace(bad.claims[0].evidence, **changes)
    return bad


@pytest.fixture
def good():
    rag = factory.offline()
    text = "El plazo de prescripción de la acción es de cuatro años. El tribunal competente es el Juzgado Tercero."
    rag.ingestor.ingest_text(text, tenant="t", doc_id="d.md", source="d.md")
    a = rag.ask("¿Cuál es el plazo de prescripción de la acción?", tenant="t")
    assert a.status is Status.VERIFIED and citation_problems(a, {"d.md": text}) == []
    return a, {"d.md": text}


def test_the_auditor_flags_a_nonexistent_citation_index(good):
    a, docs = good
    bad = copy.deepcopy(a)
    bad.claims[0].citations = [9]
    assert any("cites [9]" in p for p in citation_problems(bad, docs))


def test_the_auditor_flags_an_altered_quotation_and_wrong_offsets(good):
    a, docs = good
    altered = tampered(a, text="El plazo de prescripción es de un año.")
    assert any("is not document[" in p for p in citation_problems(altered, docs))
    moved = tampered(a, start=10_000, end=10_050)
    assert any("outside" in p for p in citation_problems(moved, docs))


def test_the_auditor_flags_an_invented_document_and_missing_or_unexpected_evidence(good):
    a, docs = good
    invented = tampered(a, doc_id="sentencia-1234.pdf")
    assert any("not among the retrieved sources" in p for p in citation_problems(invented, docs))
    stripped = copy.deepcopy(a)
    stripped.claims[0].evidence = None
    assert any("supported but has no evidence" in p for p in citation_problems(stripped, docs))
    odd = copy.deepcopy(a)
    odd.claims[0].supported = False
    assert any("unsupported but carries evidence" in p for p in citation_problems(odd, docs))


# ---- pages --------------------------------------------------------------------------------------------------------

def test_page_numbers_of_pdf_evidence_are_real_pages_of_the_document():
    pages = ["Portada del expediente sin datos relevantes.",
             "El tribunal fijó la audiencia preliminar para el 5 de mayo de 2026.",
             "Anexo con firmas y timbres del juzgado."]
    doc = load_bytes(make_pdf(pages), "exp.pdf")
    rag = factory.offline()
    rag.ingestor.ingest(doc, tenant="t", doc_id="exp.pdf", source="exp.pdf")
    a = rag.ask("¿Cuándo se fijó la audiencia preliminar?", tenant="t")
    assert a.status is Status.VERIFIED
    assert [c.evidence.page for c in a.claims] == [2]  # the audiencia is on page 2, not on the page the passage starts
    assert citation_problems(a, {"exp.pdf": doc.text}, {"exp.pdf": doc.page_starts}) == []
    wrong = tampered(a, page=3)
    assert any("not on page 3" in p for p in citation_problems(wrong, {"exp.pdf": doc.text}, {"exp.pdf": doc.page_starts}))
    ghost = tampered(a, page=40)
    assert any("does not exist" in p for p in citation_problems(ghost, {"exp.pdf": doc.text}, {"exp.pdf": doc.page_starts}))


# ---- nothing is invented ----------------------------------------------------------------------------------------

FACT = "El contrato se firmó el 1 de enero de 2020 por un plazo de cinco años ante la notario Marta Rojas."
Q = "¿Cuándo se firmó el contrato y por cuánto tiempo?"


class Says:
    def __init__(self, text):
        self.text = text

    def generate(self, question, contexts, feedback=None, *, memory=None):
        return self.text


def ask_with(llm_text):
    r = factory.offline(llm=Says(llm_text))
    r.ingestor.ingest_text(FACT, tenant="t", doc_id="contrato.md", source="contrato.md")
    return r.ask(Q, tenant="t")


@pytest.mark.parametrize("invented", [
    "El contrato se firmó el 1 de enero de 2020 por un plazo de cinco años. [7]",  # a source that does not exist
    "El contrato se firmó el 3 de marzo de 2020 por un plazo de cinco años. [1]",  # a date that is not in the document
    "El contrato se firmó el 1 de enero de 2020 ante la notario Pedro Silva. [1]",  # a person that is not named
    "El contrato se firmó el 1 de enero de 2020, según la sentencia rol 4567-2021. [1]",  # a case number from nowhere
    "Según la página 12 de la escritura, el contrato se firmó el 1 de enero de 2020. [1]",  # a page that does not exist
    "El contrato se firmó el 1 de enero de 2020 por un plazo de diez años. [1]",  # a changed figure
])
def test_invented_citations_dates_names_numbers_and_pages_do_not_become_verified_answers(invented):
    a = ask_with(invented)
    assert a.status is Status.FALLBACK


def test_an_inference_is_not_presented_as_a_stated_fact():
    """The document gives a start date and a term; the end date is arithmetic, not something the document says."""
    a = ask_with("El contrato vence el 1 de enero de 2025. [1]")
    assert a.status is Status.FALLBACK


def test_a_faithful_answer_still_passes_so_the_checks_are_not_just_refusing_everything():
    a = ask_with("El contrato se firmó el 1 de enero de 2020 por un plazo de cinco años. [1]")
    assert a.status is Status.VERIFIED and citation_problems(a, {"contrato.md": FACT}) == []


def test_partial_support_is_never_presented_as_full_proof():
    a = ask_with("El contrato se firmó el 1 de enero de 2020. [1] El notario cobró dos millones de pesos. [1]")
    assert a.status is Status.FALLBACK
    assert [c.supported for c in a.claims] == [True, False]  # the response shows which part held up
    assert a.confidence < 0.95 and "verific" in a.text.lower()


# ---- missing evidence versus a failed retrieval -----------------------------------------------------------------

def test_no_evidence_is_an_abstention_but_a_failed_backend_is_an_error():
    r = factory.offline()
    r.ingestor.ingest_text(FACT, tenant="t", doc_id="contrato.md", source="contrato.md")
    nothing = r.ask("¿Cuál es la capital de Australia?", tenant="t")
    assert nothing.status is Status.FALLBACK and nothing.reason in ("no relevant context retrieved",
                                                                    "model reported insufficient context")

    def broken(*a, **k):
        raise ConnectionError("vector database unreachable")

    r.retriever.retrieve_detailed = broken
    client = TestClient(create_app(r), raise_server_exceptions=False)
    resp = client.post("/query", headers={"X-Tenant": "t"}, json={"question": Q})
    assert resp.status_code == 500 and resp.json()["error"] == "internal_error"  # not a polite "cannot verify"
    assert "unreachable" not in resp.text


# ---- ambiguous questions --------------------------------------------------------------------------------------------

def hearings(**flags):
    r = factory.offline(replace(Settings(), **flags))
    r.ingestor.ingest_text("La audiencia preliminar del expediente Alfa se fijó para el 5 de mayo de 2026.", tenant="t",
                           doc_id="alfa.md", source="alfa.md")
    r.ingestor.ingest_text("La audiencia preliminar del expediente Bravo se fijó para el 9 de junio de 2026.", tenant="t",
                           doc_id="bravo.md", source="bravo.md")
    return r


def test_a_question_that_names_no_matter_gets_an_explicit_scope_note():
    a = hearings().ask("¿Cuándo es la audiencia preliminar?", tenant="t")
    note = a.meta["scope_note"]
    assert note["kind"] == "ambiguous_subject" and sorted(note["subjects"]) == ["Alfa", "Bravo"]
    assert "may refer to only some" in note["note"]
    assert a.to_response()["meta"]["scope_note"] == note  # the client can show it next to the answer


def test_the_clarify_policy_asks_which_matter_instead_of_answering():
    a = hearings(ambiguity_policy="clarify").ask("¿Cuándo es la audiencia preliminar?", tenant="t")
    assert a.status is Status.FALLBACK and "no indica a cuál" in a.text
    assert sorted(a.meta["clarification"]["candidates"]) == ["Alfa", "Bravo"]
    assert "5 de mayo" not in a.text and "9 de junio" not in a.text


def test_naming_the_matter_removes_the_ambiguity():
    a = hearings(ambiguity_policy="clarify").ask("¿Cuándo es la audiencia preliminar del expediente Alfa?", tenant="t")
    assert a.status is Status.VERIFIED and "5 de mayo" in a.text and "scope_note" not in a.meta


def test_the_policy_can_be_switched_off():
    assert "scope_note" not in hearings(ambiguity_policy="off").ask("¿Cuándo es la audiencia preliminar?", tenant="t").meta
    with pytest.raises(ValueError):
        Settings(ambiguity_policy="guess")
