"""RAGAS evaluation: configuration, bounded and failure-accounted execution, reproducibility records.

The judge is replaced by a fake scorer, so these run offline and deterministically. The one test that calls a real
provider is opt-in (``RAGAS_INTEGRATION=1`` plus provider keys) and is skipped otherwise.
"""
import asyncio
import json
import os

import pytest

from contextrag.eval import ragas_eval as R
from contextrag.eval.ragas_eval import (JudgeConfigError, JudgeSettings, MetricNotConfigured, Sample, ScoreError)

ENV = {"RAGAS_PROVIDER": "anthropic", "RAGAS_MODEL": "claude-sonnet-5-5", "ANTHROPIC_API_KEY": "sk-ant-SECRET-123"}
SAMPLES = [Sample("a1", "q1", "answer one", ("parent one",), "ref one"),
           Sample("a2", "q2", "answer two", ("parent two",), "ref two")]
SETTINGS = JudgeSettings(provider="anthropic", model="m", concurrency=2, timeout_seconds=5, retries=2,
                         backoff_seconds=0.0)


async def no_sleep(_):
    return None


class Scripted:
    """Returns scripted values per (metric, case). A value that is an exception is raised."""

    def __init__(self, script):
        self.script = script
        self.calls = {}

    async def score(self, metric, sample):
        key = (metric, sample.case_id)
        self.calls[key] = self.calls.get(key, 0) + 1
        item = self.script[key]
        seq = item if isinstance(item, list) else [item]
        value = seq[min(self.calls[key] - 1, len(seq) - 1)]
        if isinstance(value, BaseException):
            raise value
        return value


def run_eval(scorer, metrics=("faithfulness",), settings=SETTINGS, samples=SAMPLES):
    outcomes = asyncio.run(R.evaluate(samples, scorer, settings, metrics=metrics, sleep=no_sleep))
    return outcomes, R.summarize(outcomes, metrics)


# ---- configuration ----------------------------------------------------------------------------------

def test_settings_come_from_the_environment_and_the_key_is_never_part_of_them():
    s = JudgeSettings.from_env(ENV)
    assert (s.provider, s.model, s.concurrency, s.retries) == ("anthropic", "claude-sonnet-5-5", 4, 2)
    assert "sk-ant-SECRET-123" not in json.dumps(s.public())


def test_a_missing_key_is_named_by_variable_and_its_value_never_appears():
    env = dict(ENV)
    del env["ANTHROPIC_API_KEY"]
    with pytest.raises(JudgeConfigError, match="ANTHROPIC_API_KEY is not set"):
        JudgeSettings.from_env(env)


@pytest.mark.parametrize("override, message", [
    ({"RAGAS_PROVIDER": "gemini"}, "RAGAS_PROVIDER"),
    ({"RAGAS_PROVIDER": ""}, "RAGAS_PROVIDER"),
    ({"RAGAS_MODEL": ""}, "RAGAS_MODEL"),
    ({"RAGAS_CONCURRENCY": "0"}, "RAGAS_CONCURRENCY"),
    ({"RAGAS_CONCURRENCY": "lots"}, "invalid numeric"),
    ({"RAGAS_TIMEOUT_SECONDS": "0"}, "RAGAS_TIMEOUT_SECONDS"),
    ({"RAGAS_RETRIES": "9"}, "RAGAS_RETRIES"),
    ({"RAGAS_EMBEDDING_PROVIDER": "openai"}, "OPENAI_API_KEY"),
    ({"RAGAS_EMBEDDING_PROVIDER": "cohere"}, "RAGAS_EMBEDDING_PROVIDER"),
])
def test_invalid_judge_configuration_is_refused_before_any_work(override, message):
    with pytest.raises(JudgeConfigError, match=message):
        JudgeSettings.from_env({**ENV, **override})


# ---- scoring and accounting ---------------------------------------------------------------------------

def test_scored_pairs_produce_a_mean_over_scored_pairs_only():
    scorer = Scripted({("faithfulness", "a1"): 1.0, ("faithfulness", "a2"): 0.5})
    _, summary = run_eval(scorer)
    s = summary["faithfulness"]
    assert (s.scored, s.failed, s.not_run, s.mean, s.coverage) == (2, 0, 0, 0.75, 1.0)


def test_failed_judge_calls_are_counted_as_failures_and_never_as_zero():
    scorer = Scripted({("faithfulness", "a1"): 1.0,
                       ("faithfulness", "a2"): RuntimeError("provider 503 with secret-looking body")})
    outcomes, summary = run_eval(scorer)
    failed = [o for o in outcomes if o.error]
    assert [o.case_id for o in failed] == ["a2"] and failed[0].value is None
    s = summary["faithfulness"]
    assert (s.scored, s.failed, s.mean) == (1, 1, 1.0)  # the mean ignores the failure; it does not count it as 0
    assert s.errors == {"RuntimeError": 1}
    assert "secret-looking" not in json.dumps([o.__dict__ for o in outcomes])  # messages are not recorded


def test_when_nothing_is_scored_the_mean_is_none_not_zero():
    scorer = Scripted({("faithfulness", "a1"): RuntimeError("x"), ("faithfulness", "a2"): RuntimeError("y")})
    _, summary = run_eval(scorer)
    assert summary["faithfulness"].mean is None and summary["faithfulness"].scored == 0


def test_transient_failures_are_retried_within_the_budget():
    scorer = Scripted({("faithfulness", "a1"): [RuntimeError("blip"), RuntimeError("blip"), 0.9],
                       ("faithfulness", "a2"): 0.1})
    outcomes, summary = run_eval(scorer)
    first = next(o for o in outcomes if o.case_id == "a1")
    assert first.error is None and first.attempts == 3 and first.value == 0.9
    assert summary["faithfulness"].scored == 2


def test_retries_are_bounded_and_the_last_error_is_reported():
    scorer = Scripted({("faithfulness", "a1"): RuntimeError("down"), ("faithfulness", "a2"): 1.0})
    outcomes, _ = run_eval(scorer)
    failed = next(o for o in outcomes if o.case_id == "a1")
    assert failed.attempts == SETTINGS.retries + 1 and failed.error == "RuntimeError"


def test_invalid_input_is_not_retried():
    scorer = Scripted({("faithfulness", "a1"): ValueError("response is missing"), ("faithfulness", "a2"): 1.0})
    outcomes, _ = run_eval(scorer)
    assert next(o for o in outcomes if o.case_id == "a1").attempts == 1


def test_a_timeout_is_recorded_as_a_timeout():
    class Slow:
        async def score(self, metric, sample):
            await asyncio.sleep(2.5)
            return 1.0

    settings = JudgeSettings(provider="anthropic", model="m", timeout_seconds=1, retries=0)  # 1 s is the minimum
    outcomes, summary = run_eval(Slow(), settings=settings, samples=SAMPLES[:1])
    assert outcomes[0].error == "TimeoutError" and summary["faithfulness"].errors == {"TimeoutError": 1}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.1])
def test_scores_outside_the_unit_interval_are_errors_not_values(bad):
    scorer = Scripted({("faithfulness", "a1"): bad, ("faithfulness", "a2"): 1.0})
    outcomes, _ = run_eval(scorer, settings=JudgeSettings(provider="anthropic", model="m", retries=0))
    assert next(o for o in outcomes if o.case_id == "a1").error == ScoreError.__name__


def test_a_metric_without_its_dependency_is_not_run_and_does_not_fail_the_run():
    class NoEmbeddings:
        async def score(self, metric, sample):
            if metric == "answer_relevancy":
                raise MetricNotConfigured(metric)
            return 1.0

    outcomes, summary = run_eval(NoEmbeddings(), metrics=("faithfulness", "answer_relevancy"))
    assert summary["answer_relevancy"].not_run == 2 and summary["answer_relevancy"].mean is None
    report = R.EvalReport(run={}, metrics=summary, outcomes=outcomes)
    assert report.ok()  # not run by configuration is not a failure
    assert summary["faithfulness"].scored == 2


def test_concurrency_is_bounded():
    in_flight = {"now": 0, "max": 0}

    class Tracking:
        async def score(self, metric, sample):
            in_flight["now"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["now"])
            await asyncio.sleep(0.01)
            in_flight["now"] -= 1
            return 0.5

    many = [Sample(f"c{i}", "q", "a", ("p",), "r") for i in range(12)]
    settings = JudgeSettings(provider="anthropic", model="m", concurrency=3, timeout_seconds=5, retries=0)
    run_eval(Tracking(), metrics=("faithfulness",), settings=settings, samples=many)
    assert in_flight["max"] <= 3


def test_unknown_metrics_are_rejected():
    with pytest.raises(ValueError, match="unknown metrics"):
        asyncio.run(R.evaluate(SAMPLES, Scripted({}), SETTINGS, metrics=("bleu",), sleep=no_sleep))


# ---- reproducibility and reports --------------------------------------------------------------------

def test_the_run_record_pins_the_dataset_and_the_judge_and_holds_no_secret(tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text('{"cases": []}', encoding="utf-8")

    class Constant:
        async def score(self, metric, sample):
            return 1.0

    report = R.run(SAMPLES, Constant(), JudgeSettings.from_env(ENV), dataset=dataset,
                   pipeline={"parent_tokens": 1024, "child_tokens": 256}, reference_source="reference",
                   metrics=("faithfulness",))
    rec = report.run
    assert rec["dataset"]["sha256"] == R.dataset_sha256(dataset)
    assert rec["judge"]["model"] == "claude-sonnet-5-5"
    assert rec["pipeline"] == {"parent_tokens": 1024, "child_tokens": 256}
    assert rec["dataset"]["reference_source"] == "reference"
    out = tmp_path / "report.json"
    R.write_report(report, out)
    text = out.read_text(encoding="utf-8")
    assert "sk-ant-SECRET-123" not in text
    assert json.loads(text)["metrics"]["faithfulness"]["coverage"] == 1.0


def test_samples_keep_the_raw_parent_passages_and_say_when_the_reference_is_weak(tmp_path):
    from contextrag.factory import offline

    rag = offline()
    rag.ingestor.ingest_text("Los registros de auditoría se conservan durante siete años completos. " * 3,
                             tenant="t", doc_id="politica.md", source="politica.md")
    cases = [{"id": "c1", "question": "¿Cuánto tiempo se conservan los registros de auditoría?", "expect": "siete años"}]
    samples, source = R.samples_from_rag(rag, cases, tenant="t")
    assert source == "expect"  # only a short expected phrase: recorded, so the weak reference is visible
    assert samples[0].contexts and all(isinstance(c, str) and c for c in samples[0].contexts)
    assert samples[0].response
    rag.close()


def test_a_case_without_any_reference_is_refused():
    class Rag:
        def ask(self, *a, **k):
            raise AssertionError("must not be called")

    with pytest.raises(JudgeConfigError, match="no reference"):
        R.samples_from_rag(Rag(), [{"id": "c1", "question": "q"}], tenant="t")


# ---- opt-in live judge ----------------------------------------------------------------------------------

@pytest.mark.skipif(os.getenv("RAGAS_INTEGRATION") != "1", reason="set RAGAS_INTEGRATION=1 and provider keys to run")
def test_live_judge_scores_a_faithful_answer_higher_than_a_hallucinated_one():
    settings = JudgeSettings.from_env()  # RAGAS_PROVIDER, RAGAS_MODEL and the provider key from the environment
    scorer = R.RagasScorer(settings)
    faithful = Sample("f", "¿Cuánto se conservan los registros?", "Se conservan siete años.",
                      ("Los registros de auditoría se conservan siete años.",), "siete años")
    invented = Sample("h", "¿Cuánto se conservan los registros?", "Se conservan para siempre en Marte.",
                      ("Los registros de auditoría se conservan siete años.",), "siete años")

    async def both():
        return (await scorer.score("faithfulness", faithful), await scorer.score("faithfulness", invented))

    good, bad = asyncio.run(both())
    assert good > bad
