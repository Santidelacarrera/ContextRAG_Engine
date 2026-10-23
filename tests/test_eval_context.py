"""Retrieval-side RAG metrics: hand-checked values, deterministic corpus, and the recorded regression baseline."""
import json
from pathlib import Path

from contextrag import factory
from contextrag.core.models import Context, ParentChunk
from contextrag.eval.context_metrics import (average_precision, context_recall, evaluate_context, regressions)
from contextrag.eval.harness import load_json
from contextrag.eval.long_corpus import build_long_corpus

BASELINE = Path(__file__).resolve().parents[1] / "contextrag" / "eval" / "data" / "context_baseline.json"


def passage(text):
    return Context(1, ParentChunk(id=text[:8], doc_id="d", tenant="t", source="d", text=text, start=0, end=len(text)), 1.0)


def test_recall_is_the_share_of_reference_terms_present_in_the_evidence():
    evidence = [passage("Los registros se conservan durante siete años."), passage("Cifrado AES.")]
    assert context_recall("siete años", evidence) == 1.0
    # four reference terms (siete, años, cifrado, kerberos); three appear in the evidence, kerberos does not
    assert context_recall("siete años y cifrado de Kerberos", evidence) == 0.75


def test_average_precision_rewards_the_relevant_passage_being_early():
    relevant = passage("El plazo es de siete años.")
    noise = passage("Nada que ver con el tema.")
    assert average_precision("siete años", [relevant, noise]) == 1.0
    assert average_precision("siete años", [noise, relevant]) == 0.5  # first hit at position 2
    assert average_precision("siete años", [noise]) == 0.0


def test_the_synthetic_corpus_is_deterministic():
    assert build_long_corpus() == build_long_corpus()
    assert build_long_corpus(seed=8) != build_long_corpus(seed=7)


def test_context_metrics_on_the_bundled_dataset_are_perfect_and_say_little():
    report = evaluate_context(factory.offline, load_json("rag_cases.json"))
    assert report.total == 5 and report.mean_recall == 1.0  # saturated: too small to separate configurations


def test_no_regression_against_the_recorded_baseline_on_the_synthetic_corpus():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))["metrics"]
    report = evaluate_context(factory.offline, build_long_corpus())
    assert regressions(report, baseline, tolerance=0.0) == []


def test_regressions_are_reported_when_a_metric_drops():
    class R:
        mean_recall, mean_average_precision = 0.5, 0.9
    out = regressions(R(), {"mean_recall": 0.9, "mean_average_precision": 0.8})
    assert out == ["mean_recall: 0.5000 < baseline 0.9000"]
