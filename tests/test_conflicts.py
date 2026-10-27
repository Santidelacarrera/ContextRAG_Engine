"""Conflicts between sources: explicit contradictions, complementary information and ambiguous cases.

The structured detector is a heuristic over dates and figures. These tests state what it finds and, as plainly, what it
cannot see. The NLI detector is exercised with a stand-in model, which proves the plumbing and nothing about detection
quality with a real model.
"""
from dataclasses import replace

import pytest

from conftest import make_pdf
from contextrag import Status, factory
from contextrag.core.config import Settings
from contextrag.core.ingestion import load_bytes
from contextrag.verification.conflicts import (NLIConflictDetector, StructuredConflictDetector, _dates, _numbers,
                                               build_detector)
from contextrag.verification.nli_verifier import LexicalEntailment

HEARING_A = "La audiencia preliminar del expediente Alfa se fijó para el 5 de mayo de 2026."
HEARING_B = "La audiencia preliminar del expediente Alfa se fijó para el 9 de junio de 2026."


# ---- normalisation -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("el 5 de mayo de 2026", {"2026-05-05"}), ("el 05/05/2026", {"2026-05-05"}), ("on 2026-05-05", {"2026-05-05"}),
    ("5 May 2026", {"2026-05-05"}), ("May 5, 2026", {"2026-05-05"}), ("el 9 de junio de 2026 y el 5 de mayo de 2026",
                                                                   {"2026-06-09", "2026-05-05"}),
])
def test_dates_are_normalised_across_formats(text, expected):
    assert _dates(text)[0] == frozenset(expected)


def test_dates_are_removed_before_numbers_are_read_and_amounts_are_normalised():
    dates, rest = _dates("La audiencia es el 5 de mayo de 2026 por 120 millones.")
    assert _numbers(rest) == frozenset({"1.2e+08"})
    assert _numbers("120.000.000 de pesos") == _numbers("120 millones de pesos")
    assert _numbers("3,5 mil") == frozenset({"3500"})


# ---- the structured detector over sentence pairs ---------------------------------------------------------------

def detect(a, b, doc_a="a.md", doc_b="b.md", question="¿Cuándo es la audiencia preliminar del expediente Alfa?"):
    r = factory.offline(replace(Settings(), conflict_detection="off"))
    r.ingestor.ingest_text(a, tenant="t", doc_id=doc_a, source=doc_a)
    r.ingestor.ingest_text(b, tenant="t", doc_id=doc_b, source=doc_b)
    _, _, retrieved = r._prepare(question, "t", frozenset(), None)  # what the pipeline hands the detector
    return StructuredConflictDetector().detect(question, retrieved)


def test_two_different_dates_for_the_same_event_are_a_conflict_with_both_sources_located():
    found = detect(HEARING_A, HEARING_B)
    assert len(found) == 1
    c = found[0]
    assert c.kind == "value_mismatch" and c.needs_human_review and "heuristic" in c.method
    assert {c.a.doc_id, c.b.doc_id} == {"a.md", "b.md"}
    for ref in (c.a, c.b):  # offsets point at the sentence inside its document
        assert ref.text in (HEARING_A, HEARING_B) and ref.end - ref.start == len(ref.text)


def test_the_same_date_written_two_ways_is_not_a_conflict():
    assert detect(HEARING_A, "La audiencia preliminar del expediente Alfa se fijó para el 05/05/2026.") == []


def test_extra_detail_is_not_a_conflict():
    assert detect(HEARING_A, HEARING_A[:-1] + " a las 10:00 horas en la sala dos.") == []


def test_complementary_information_about_different_facts_is_not_a_conflict():
    signing = "El contrato del expediente Alfa se firmó el 5 de mayo de 2026."
    expiry = "El contrato del expediente Alfa vence el 5 de mayo de 2029."
    assert detect(signing, expiry, question="¿Qué fechas tiene el contrato del expediente Alfa?") == []


def test_the_same_wording_about_different_subjects_is_not_a_conflict():
    other = "La audiencia preliminar del expediente Bravo se fijó para el 9 de junio de 2026."
    assert detect(HEARING_A, other, question="¿Cuándo es la audiencia preliminar?") == []


def test_different_figures_for_the_same_statement_conflict_but_equal_amounts_do_not():
    a = "El capital social de Industrias Valdivia S.A. es de 120 millones de pesos."
    assert detect(a, a.replace("120 millones", "150 millones"), question="¿Cuál es el capital social de Industrias Valdivia S.A.?")
    assert detect(a, a.replace("120 millones", "120.000.000"), question="¿Cuál es el capital social de Industrias Valdivia S.A.?") == []


def test_one_source_never_conflicts_with_itself_on_the_same_page():
    assert detect(HEARING_A + " " + HEARING_B, "Otro texto sin relación alguna.") == []


# ---- labelled pairs: what the heuristic catches and what it cannot -------------------------------------------------

LABELLED = [  # (id, a, b, a real contradiction?)
    ("date", HEARING_A, HEARING_B, True),
    ("amount", "El monto de la indemnización es de 80 millones.", "El monto de la indemnización es de 95 millones.", True),
    ("same", HEARING_A, HEARING_A, False),
    ("format", HEARING_A, "La audiencia preliminar del expediente Alfa se fijó para el 05/05/2026.", False),
    ("detail", HEARING_A, HEARING_A[:-1] + " a las 10:00.", False),
    ("subjects", HEARING_A, HEARING_B.replace("Alfa", "Bravo"), False),
    ("facts", "El contrato Alfa se firmó el 5 de mayo de 2026.", "El contrato Alfa vence el 5 de mayo de 2029.", False),
    # contradictions the heuristic cannot see: no figures, or the same figure with opposite meaning
    ("negation", "La sentencia del expediente Alfa fue notificada a la demandada.",
     "La sentencia del expediente Alfa no fue notificada a la demandada.", True),
    ("status", "El contrato Alfa se encuentra vigente al día de hoy.", "El contrato Alfa se encuentra terminado al día de hoy.", True),
    ("party", "El demandante del expediente Alfa es Pedro Soto.", "El demandante del expediente Alfa es Juan Pérez.", True),
]


def test_measured_behaviour_of_the_structured_detector_on_labelled_pairs():
    flagged = {}
    for case_id, a, b, _ in LABELLED:
        flagged[case_id] = bool(StructuredConflictDetector._score(a, b))
    truth = {case_id: real for case_id, _, _, real in LABELLED}
    tp = sum(flagged[k] and truth[k] for k in truth)
    fp = sum(flagged[k] and not truth[k] for k in truth)
    fn = sum(not flagged[k] and truth[k] for k in truth)
    # Measured after figures written as words and negation flips were added (see tests/test_conflict_bench.py for the
    # category breakdown). The earlier expectation (2, 0, 3) was recorded before those rules existed; it is updated, not hidden.
    assert (tp, fp, fn) == (3, 0, 2)
    assert sorted(k for k in truth if truth[k] and not flagged[k]) == ["party", "status"]  # still-documented blind spots


# ---- in the pipeline ---------------------------------------------------------------------------------------------

def rag(**flags):
    r = factory.offline(replace(Settings(), **flags))
    r.ingestor.ingest_text(HEARING_A, tenant="t", doc_id="a.md", source="a.md")
    r.ingestor.ingest_text(HEARING_B, tenant="t", doc_id="b.md", source="b.md")
    return r


Q = "¿Cuándo es la audiencia preliminar del expediente Alfa?"


def test_conflicting_sources_produce_no_answer_but_both_sources_and_a_review_flag():
    r = rag()
    a = r.ask(Q, tenant="t")
    assert a.status is Status.FALLBACK and a.reason == "conflicting sources: human review needed"
    assert "5 de mayo" not in a.text and "9 de junio" not in a.text  # neither version is presented as the answer
    body = a.to_response()
    assert body["meta"]["needs_human_review"] is True
    sources = body["meta"]["conflicts"][0]["sources"]
    assert {s["doc_id"] for s in sources} == {"a.md", "b.md"} and all(s["start"] < s["end"] for s in sources)
    assert "conflicts_detected_total 1" in r.metrics_text()


def test_without_a_conflict_the_same_question_is_answered():
    r = factory.offline()
    r.ingestor.ingest_text(HEARING_A, tenant="t", doc_id="a.md", source="a.md")
    r.ingestor.ingest_text("El Juzgado Tercero tiene su sede en Temuco.", tenant="t", doc_id="b.md", source="b.md")
    assert r.ask(Q, tenant="t").status is Status.VERIFIED


def test_with_detection_off_the_conflict_is_not_reported():
    """Documents what turning the control off means: the system may answer from either source."""
    a = rag(conflict_detection="off").ask(Q, tenant="t")
    assert "conflicts" not in a.meta


def test_a_conflict_with_a_document_the_caller_cannot_read_is_neither_detected_nor_revealed():
    r = factory.offline()
    r.ingestor.ingest_text(HEARING_A, tenant="t", doc_id="a.md", source="a.md")
    r.ingestor.ingest_text(HEARING_B, tenant="t", doc_id="secret.md", source="secret.md", acl=("caso-x",))
    visible = r.ask(Q, tenant="t", groups=frozenset())
    assert visible.status is Status.VERIFIED and "5 de mayo" in visible.text
    assert "secret.md" not in str(visible.to_response()) and "9 de junio" not in str(visible.to_response())
    assert r.ask(Q, tenant="t", groups=frozenset({"caso-x"})).status is Status.FALLBACK  # the authorised reader sees it


def test_page_numbers_are_kept_for_pdf_sources():
    r = factory.offline(replace(Settings(), conflict_detection="structured"))
    pdf_a = make_pdf(["Portada del expediente.", HEARING_A])
    pdf_b = make_pdf([HEARING_B])
    r.ingestor.ingest(load_bytes(pdf_a, "a.pdf"), tenant="t", doc_id="a.pdf", source="a.pdf")
    r.ingestor.ingest(load_bytes(pdf_b, "b.pdf"), tenant="t", doc_id="b.pdf", source="b.pdf")
    a = r.ask(Q, tenant="t")
    assert a.status is Status.FALLBACK
    pages = {s["doc_id"]: s["page"] for s in a.meta["conflicts"][0]["sources"]}
    assert pages == {"a.pdf": 2, "b.pdf": 1}


# ---- the NLI detector, with a stand-in model ------------------------------------------------------------------------

class Scripted:
    """Stand-in for an NLI model with a contradiction class: contradiction when the two texts differ in one marker."""

    def __init__(self, hot=0.97):
        self.hot, self.calls = hot, 0

    def contradiction(self, pairs):
        self.calls += 1
        return [self.hot if (" no " in f" {a} ") != (" no " in f" {b} ") else 0.02 for a, b in pairs]


def test_the_nli_detector_finds_a_contradiction_the_structured_one_cannot_see():
    """Status changes ("vigente" against "terminado") are not figures or negations, so only a semantic detector can see them."""
    r = factory.offline(replace(Settings(), conflict_detection="off"))
    for doc, text in (("a.md", "El contrato Epsilon se encuentra vigente al día de hoy."),
                      ("b.md", "El contrato Epsilon se encuentra terminado al día de hoy.")):
        r.ingestor.ingest_text(text, tenant="t", doc_id=doc, source=doc)
    q = "¿Está vigente el contrato Epsilon?"
    _, _, retrieved = r._prepare(q, "t", frozenset(), None)
    assert StructuredConflictDetector().detect(q, retrieved) == []
    class StatusAware:
        """Stand-in with the behaviour a contradiction head would show on this pair (a test double, not a model)."""

        def contradiction(self, pairs):
            return [0.97 if {"vigente", "terminado"} <= set(a.lower().split() + b.lower().split()) else 0.02 for a, b in pairs]

    found = NLIConflictDetector(StatusAware()).detect(q, retrieved)
    assert found and found[0].kind == "contradiction" and found[0].score == pytest.approx(0.97)
    assert "uncalibrated" in found[0].method

def test_the_nli_detector_respects_its_threshold_and_can_be_injected_into_the_pipeline():
    model = Scripted(hot=0.5)
    r = factory.offline(replace(Settings(), conflict_detection="off"))
    r.conflict_detector = NLIConflictDetector(model, threshold=0.9, name="stand-in-cold")
    for doc, text in (("a.md", "La sentencia del expediente Alfa fue notificada a la demandada."),
                      ("b.md", "La sentencia del expediente Alfa no fue notificada a la demandada.")):
        r.ingestor.ingest_text(text, tenant="t", doc_id=doc, source=doc)
    assert r.ask("¿Fue notificada la sentencia del expediente Alfa?", tenant="t").meta.get("conflicts") is None
    r.conflict_detector = NLIConflictDetector(Scripted(hot=0.97), threshold=0.9, name="stand-in-hot")
    answer = r.ask("¿Fue notificada la sentencia del expediente Alfa?", tenant="t")
    assert answer.status is Status.FALLBACK and answer.meta["conflicts"][0]["kind"] == "contradiction"


def test_nli_mode_refuses_a_model_that_cannot_detect_contradiction():
    with pytest.raises(ValueError, match="no 'contradiction' method"):
        build_detector("nli", LexicalEntailment(), 0.9)
    with pytest.raises(ValueError, match="no 'contradiction' method"):
        factory.offline(replace(Settings(), conflict_detection="nli"))  # fails at construction, never silently
    assert build_detector("off", LexicalEntailment(), 0.9) is None


def test_the_configuration_rejects_unknown_modes_and_thresholds():
    with pytest.raises(ValueError):
        Settings(conflict_detection="fuzzy")
    with pytest.raises(ValueError):
        Settings(conflict_threshold=0.0)


# ---- the answer cache must not outlive the models that produced the answer ------------------------------------------

def test_the_cache_key_changes_with_the_nli_model_the_reranker_and_the_conflict_detector():
    from contextrag.retrieval.reranker import BoundedReranker

    class Named:
        def __init__(self, name):
            self.model_name = name

        def score(self, query, passages):
            return [1.0] * len(passages)

    def key(rerank_name="m1", nli_name="n1", detector=None):
        r = factory.offline(replace(Settings(), conflict_detection="off"))
        r.reranker = BoundedReranker(Named(rerank_name), 5)
        r.verifier.nli = type("N", (), {"model_name": nli_name, "entailment": lambda self, p: [1.0] * len(p)})()
        r.conflict_detector = detector
        return r._key("q", "t", frozenset(), None)

    base = key()
    assert key() == base  # stable for identical components
    assert key(rerank_name="m2") != base and key(nli_name="n2") != base
    assert key(detector=StructuredConflictDetector()) != base
