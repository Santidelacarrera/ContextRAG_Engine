"""Entity benchmark: dataset integrity, the measured rates, and the cases the design is known not to cover."""
import json

import pytest

from contextrag.eval import entity_bench as E
from contextrag.eval.entity_corpus import DATA_PATH, build_entity_bench


@pytest.fixture(scope="module")
def report():
    return E.run_entity_bench()


def test_the_checked_in_dataset_is_what_the_generator_produces():
    assert DATA_PATH.read_text(encoding="utf-8") == json.dumps(build_entity_bench(), indent=1, ensure_ascii=False) + "\n"
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    assert data["synthetic"] is True and len(data["queries"]) >= 40
    kinds = {q["kind"] for q in data["queries"]}
    assert {"canonical", "alias_other_doc", "alias_same_doc", "unknown", "decoy", "ambiguous", "restricted_denied"} <= kinds
    assert {q["expected"] for q in data["queries"]} == {"answer", "abstain", "clarify"}


def test_the_default_configuration_has_no_false_answers_and_no_false_abstentions_on_the_headline_set(report):
    m = report["configs"]["alias_resolution"]["metrics"]
    assert m["false_answer_rate"] == 0.0 and m["false_abstention_rate"] == 0.0
    assert m["wrong_answer_rate"] == 0.0 and m["answered_correctly_rate"] == 1.0
    assert m["correct_abstention_rate"] == 1.0 and m["correct_clarification_rate"] == 1.0
    assert report["failures"]["alias_resolution"] == []


def test_alias_resolution_removes_false_abstentions_without_adding_false_answers(report):
    gate, resolved = (report["configs"][k]["metrics"] for k in ("gate_only", "alias_resolution"))
    assert resolved["false_abstention_rate"] < gate["false_abstention_rate"]
    assert resolved["false_answer_rate"] <= gate["false_answer_rate"]  # never traded safety for answers


def test_the_gate_matters_without_it_most_unanswerable_questions_are_answered(report):
    assert report["configs"]["no_gate"]["metrics"]["false_answer_rate"] > 0.5


def test_false_answers_and_false_abstentions_are_separate_metrics(report):
    m = report["configs"]["alias_resolution"]["metrics"]
    assert {"false_answer_rate", "false_abstention_rate", "wrong_answer_rate"} <= set(m)


def test_known_limits_are_reported_apart_and_remain_unresolved(report):
    """If one of these starts to pass, the design changed: update the documentation of the limit, then this test."""
    limits = {x["kind"] + ":" + x["id"]: (x["expected"], x["observed"])
              for x in report["configs"]["alias_resolution"]["known_limits"]}
    assert limits["limit_alias_form:lim43"] == ("answer", "abstain")  # alias written in a form no pattern reads
    assert limits["limit_homonym:lim44"] == ("clarify", "answer")  # undeclared homonym: nothing to detect
    assert limits["limit_attribute:lim46"] == ("abstain", "answer")  # entity exists, attribute does not
    assert all(not k.startswith("limit_") for k in report["failures"])  # and they never enter the headline


def test_the_report_states_that_it_is_offline_and_synthetic(report):
    assert "synthetic" in report["caveat"] and "not quality with real models" in report["caveat"]
