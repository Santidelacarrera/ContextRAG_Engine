"""Entity coverage: a faithful answer about the wrong subject must not be returned as a verified answer."""
from dataclasses import replace

import pytest

from contextrag import Status, factory
from contextrag.core.config import Settings
from contextrag.verification.entities import missing_terms, named_terms

DOC = ("# Zelrox runbook\n\nZelrox is owned by the Payments team.\n\n## Data retention\n\n"
       "The retention period for Zelrox logs is 90 days. After that the records are deleted automatically.\n\n"
       "## Security\n\nZelrox encrypts data at rest using AES-256.\n")


# ---- extraction ---------------------------------------------------------------------------------

@pytest.mark.parametrize("question, expected", [
    ("What is the retention period for Halxanx logs?", ["Halxanx"]),
    ("Which team owns Zelrox?", ["Zelrox"]),
    ("What is the capital of Atlantis?", ["Atlantis"]),
    ("Does the service use AES-256 at rest?", ["AES-256"]),
    ("¿Cuánto tiempo se conservan los registros de Nexus?", ["Nexus"]),
    ("What is the retention period?", []),  # the first word is never a name
    ("I have a problem. How long are logs kept?", []),  # nor is the first word of a later sentence
    ("how long does the cipher keep logs", []),  # lowercase words are never names
    ("Is the follow-the-sun rotation documented?", []),  # a hyphenated lowercase word is not an identifier
    ("Compare Alpha and Beta, and Alpha again", ["Alpha", "Beta"]),  # no duplicates, order kept
])
def test_named_terms(question, expected):
    assert named_terms(question) == expected


def test_a_term_is_present_when_every_part_of_it_appears_in_the_evidence():
    assert missing_terms("Does Zelrox use AES-256?", [DOC]) == []
    assert missing_terms("Does Zelrox use AES-128?", [DOC]) == ["AES-128"]  # 128 is not in the evidence
    assert missing_terms("Where does Zelrox run? Also Foobar.", [DOC]) == ["Foobar"]


def test_matching_ignores_case_and_plural_forms():
    assert missing_terms("What do the Services do?", ["each service runs on its own"]) == []
    assert missing_terms("Tell me about ZELROX", [DOC]) == []


# ---- pipeline -----------------------------------------------------------------------------------

class CountingLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, question, contexts, feedback=None, *, memory=None):
        self.calls += 1
        return "The retention period for Zelrox logs is 90 days. [1]"


def rag(flag=True, llm=None):
    r = factory.offline(replace(Settings(), require_question_entities=flag), llm=llm)
    r.ingestor.ingest_text(DOC, tenant="t", doc_id="zelrox.md", source="zelrox.md")
    return r


def test_a_question_about_an_unindexed_name_abstains_without_calling_the_model():
    llm = CountingLLM()
    a = rag(llm=llm).ask("What is the retention period for Halxanx logs?", tenant="t")
    assert a.status is Status.FALLBACK
    assert "named term" in a.reason
    assert Halxanx_not_echoed(a)
    assert llm.calls == 0  # no model call, so no cost and no chance of a plausible but wrong answer


def Halxanx_not_echoed(answer):
    return "Halxanx" not in answer.reason  # the reason names a count of the problem, not the user's term


def test_a_question_about_an_indexed_name_is_still_answered():
    llm = CountingLLM()
    a = rag(llm=llm).ask("What is the retention period for Zelrox logs?", tenant="t")
    assert a.status is Status.VERIFIED and "90 days" in a.text and llm.calls == 1


def test_without_the_gate_the_wrong_subject_answer_is_returned_as_verified():
    """Documents the failure the gate exists for: every statement is faithful to its passage, none is about Halxanx."""
    a = rag(flag=False, llm=CountingLLM()).ask("What is the retention period for Halxanx logs?", tenant="t")
    assert a.status is Status.VERIFIED and "Zelrox" in a.text


def test_a_question_that_names_nothing_is_not_blocked():
    a = rag().ask("What is the retention period for these logs?", tenant="t")
    assert a.status is Status.VERIFIED


def test_the_gate_is_counted_and_part_of_the_cache_key():
    r = rag()
    r.ask("What is the retention period for Halxanx logs?", tenant="t")
    assert "entity_gate_rejections_total 1" in r.metrics_text()
    other = factory.offline(replace(Settings(), require_question_entities=False))
    assert r._key("q", "t", frozenset(), None) != other._key("q", "t", frozenset(), None)


@pytest.mark.parametrize("raw, expected", [("false", False), ("0", False), ("No", False), ("OFF", False),
                                           ("true", True), ("1", True), ("yes", True), (" On ", True)])
def test_the_flag_is_parsed_as_a_boolean_not_as_a_non_empty_string(monkeypatch, raw, expected):
    monkeypatch.setenv("CONTEXTRAG_REQUIRE_QUESTION_ENTITIES", raw)
    assert Settings.from_env().require_question_entities is expected


@pytest.mark.parametrize("raw", ["", "maybe", "2", "disable"])
def test_an_ambiguous_boolean_is_refused_rather_than_guessed(monkeypatch, raw):
    monkeypatch.setenv("CONTEXTRAG_REQUIRE_QUESTION_ENTITIES", raw)
    with pytest.raises(ValueError, match="expected a boolean"):
        Settings.from_env()


# ---- entity phrases and the claim-level gate ---------------------------------------------------------

from contextrag.verification.entities import named_phrases  # noqa: E402


@pytest.mark.parametrize("question, expected", [
    ("¿Cuál es el capital social de Industrias Valdivia S.A.?", ["Industrias Valdivia S.A"]),
    ("¿Qué dice el Código de Procedimiento Civil sobre notificaciones?", ["Código de Procedimiento Civil"]),
    ("Compare Alfa S.A. y Beta Ltda. en Talca", ["Alfa S.A", "Beta Ltda", "Talca"]),  # "y" separates two entities
    ("¿Dónde tiene su domicilio SR?", ["SR"]),
    ("¿Cuál es el plazo?", []),
])
def test_named_phrases(question, expected):
    assert named_phrases(question) == expected


def test_a_true_statement_about_another_entity_is_not_part_of_the_answer():
    """Each sentence is faithful to its own document, but only one of them is about the entity that was asked about."""
    class Mixed:
        def generate(self, question, contexts, feedback=None, *, memory=None):
            return ("Alfa Ltda. tiene su domicilio en Talca. [1] Beta Ltda. tiene su domicilio en Arica. [2]")

    r = factory.offline(llm=Mixed())
    r.ingestor.ingest_text("Alfa Ltda. tiene su domicilio en Talca.", tenant="t", doc_id="alfa.md", source="alfa.md")
    r.ingestor.ingest_text("Beta Ltda. tiene su domicilio en Arica.", tenant="t", doc_id="beta.md", source="beta.md")
    a = r.ask("¿Dónde tiene su domicilio Alfa Ltda.?", tenant="t")
    assert a.status is Status.FALLBACK  # the unrelated claim is rejected, so the answer is not presented as verified
    assert "entity_consistency_rejections_total" in r.metrics_text()


def test_a_question_about_two_entities_accepts_claims_about_either():
    class Both:
        def generate(self, question, contexts, feedback=None, *, memory=None):
            return "Alfa Ltda. tiene su domicilio en Talca. [1] Beta Ltda. tiene su domicilio en Arica. [2]"

    r = factory.offline(llm=Both())
    r.ingestor.ingest_text("Alfa Ltda. tiene su domicilio en Talca.", tenant="t", doc_id="alfa.md", source="alfa.md")
    r.ingestor.ingest_text("Beta Ltda. tiene su domicilio en Arica.", tenant="t", doc_id="beta.md", source="beta.md")
    a = r.ask("¿Dónde tienen su domicilio Alfa Ltda. y Beta Ltda.?", tenant="t")
    assert a.status is Status.VERIFIED
