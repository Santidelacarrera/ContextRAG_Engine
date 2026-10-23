"""LLM-as-judge evaluation of the RAG pipeline with RAGAS metrics.

Three metrics, all from RAGAS' ``collections`` API (ragas 0.4):

* ``faithfulness``: are the answer's claims supported by the raw parent passages the model was given? The passages
  are the full parent chunks (``Context.parent.text``), not the span-compressed text, so the judge sees the same
  material a reader of the source would.
* ``answer_relevancy``: does the answer address the question? Needs an embedding model, so it runs only when
  ``RAGAS_EMBEDDING_PROVIDER`` is configured; otherwise it is reported as *not run*, never as zero.
* ``context_recall``: does the retrieved evidence contain what the reference answer says?

Failure accounting (the rule this module is built around): every (case, metric) pair ends as exactly one of
*scored*, *failed* (with an error class name) or *not run*. A mean is taken over scored pairs only, and it is ``None``
when nothing was scored. A failed judge call is never converted into a score of 0.

Bounded concurrency (``RAGAS_CONCURRENCY``), a per-call timeout (``RAGAS_TIMEOUT_SECONDS``) and retries with
exponential backoff (``RAGAS_RETRIES``) apply to every judge call. Provider keys come only from the environment
(``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``) and never appear in a report or a log line.

Reproducibility: a run record holds the judge and embedding model, the pipeline settings, the SHA-256 of the dataset
file, the library versions and the UTC start time. Judge output is not deterministic; the record makes that visible
rather than pretending otherwise.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

METRICS = ("faithfulness", "answer_relevancy", "context_recall")
PROVIDERS = ("anthropic", "openai")
_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


class JudgeConfigError(ValueError):
    """The judge is not configured correctly. Raised before any work starts."""


class MetricNotConfigured(RuntimeError):
    """The metric cannot run with the current configuration (for example, no embedding provider)."""


class ScoreError(RuntimeError):
    """A judge answered, but not with a usable score (NaN, infinite, or outside [0, 1])."""


@dataclass(frozen=True)
class JudgeSettings:
    provider: str
    model: str
    embedding_provider: str | None = None
    embedding_model: str | None = None
    concurrency: int = 4
    timeout_seconds: float = 60.0
    retries: int = 2
    backoff_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise JudgeConfigError(f"RAGAS_PROVIDER must be one of {PROVIDERS}")
        if not self.model:
            raise JudgeConfigError("RAGAS_MODEL is required")
        if self.embedding_provider not in (None, "openai"):
            raise JudgeConfigError("RAGAS_EMBEDDING_PROVIDER must be 'openai' or unset")
        if self.embedding_provider and not self.embedding_model:
            raise JudgeConfigError("RAGAS_EMBEDDING_MODEL is required with RAGAS_EMBEDDING_PROVIDER")
        if not 1 <= self.concurrency <= 32:
            raise JudgeConfigError("RAGAS_CONCURRENCY must be between 1 and 32")
        if not 1 <= self.timeout_seconds <= 600:
            raise JudgeConfigError("RAGAS_TIMEOUT_SECONDS must be between 1 and 600")
        if not 0 <= self.retries <= 5:
            raise JudgeConfigError("RAGAS_RETRIES must be between 0 and 5")
        if self.backoff_seconds < 0:
            raise JudgeConfigError("RAGAS_BACKOFF_SECONDS must be >= 0")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> JudgeSettings:
        e = os.environ if env is None else env
        provider = (e.get("RAGAS_PROVIDER") or "").strip().lower()
        if provider not in PROVIDERS:
            raise JudgeConfigError("set RAGAS_PROVIDER to 'anthropic' or 'openai'")
        if not (e.get(_KEY_ENV[provider]) or "").strip():
            raise JudgeConfigError(f"{_KEY_ENV[provider]} is not set")  # names the variable, never its value
        emb = (e.get("RAGAS_EMBEDDING_PROVIDER") or "").strip().lower() or None
        if emb == "openai" and not (e.get("OPENAI_API_KEY") or "").strip():
            raise JudgeConfigError("OPENAI_API_KEY is not set (needed for RAGAS_EMBEDDING_PROVIDER=openai)")
        try:
            return cls(provider=provider, model=(e.get("RAGAS_MODEL") or "").strip(),
                       embedding_provider=emb, embedding_model=(e.get("RAGAS_EMBEDDING_MODEL") or "").strip() or None,
                       concurrency=int(e.get("RAGAS_CONCURRENCY", "4")),
                       timeout_seconds=float(e.get("RAGAS_TIMEOUT_SECONDS", "60")),
                       retries=int(e.get("RAGAS_RETRIES", "2")),
                       backoff_seconds=float(e.get("RAGAS_BACKOFF_SECONDS", "1.0")))
        except ValueError as exc:
            raise JudgeConfigError(f"invalid numeric judge setting: {exc}") from None

    def public(self) -> dict[str, Any]:
        """The settings as recorded in a report. Contains no secret: keys are never part of these settings."""
        return asdict(self)


@dataclass(frozen=True)
class Sample:
    case_id: str
    question: str
    response: str
    contexts: tuple[str, ...]  # raw parent passages, in the order the model received them
    reference: str


class Scorer(Protocol):
    async def score(self, metric: str, sample: Sample) -> float: ...


@dataclass(frozen=True)
class Outcome:
    metric: str
    case_id: str
    value: float | None
    error: str | None  # None when scored; otherwise an error class name or "not_configured"
    attempts: int


@dataclass
class MetricSummary:
    metric: str
    total: int
    scored: int = 0
    failed: int = 0
    not_run: int = 0
    mean: float | None = None  # over scored pairs only; None when nothing was scored
    errors: dict[str, int] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return self.scored / self.total if self.total else 0.0


@dataclass
class EvalReport:
    run: dict[str, Any]
    metrics: dict[str, MetricSummary]
    outcomes: list[Outcome]

    def ok(self) -> bool:
        """True when every configured pair was scored. A metric that is *not run* by configuration is not a failure."""
        return all(m.failed == 0 for m in self.metrics.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "metrics": {k: {**asdict(v), "coverage": v.coverage} for k, v in self.metrics.items()},
            "outcomes": [asdict(o) for o in self.outcomes],
        }


def summarize(outcomes: Sequence[Outcome], metrics: Sequence[str]) -> dict[str, MetricSummary]:
    out: dict[str, MetricSummary] = {}
    for name in metrics:
        mine = [o for o in outcomes if o.metric == name]
        s = MetricSummary(metric=name, total=len(mine))
        values = []
        for o in mine:
            if o.error is None:
                s.scored += 1
                values.append(o.value)
            elif o.error == "not_configured":
                s.not_run += 1
            else:
                s.failed += 1
                s.errors[o.error] = s.errors.get(o.error, 0) + 1
        s.mean = sum(v for v in values if v is not None) / len(values) if values else None
        out[name] = s
    return out


async def evaluate(samples: Sequence[Sample], scorer: Scorer, settings: JudgeSettings, *,
                   metrics: Sequence[str] = METRICS, sleep: Callable[[float], Any] = asyncio.sleep) -> list[Outcome]:
    """Score every (sample, metric) pair with bounded concurrency. Never raises for a judge failure: it is recorded."""
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    gate = asyncio.Semaphore(settings.concurrency)

    async def one(sample: Sample, metric: str) -> Outcome:
        error: str | None = None
        attempts = 0
        async with gate:
            for attempt in range(settings.retries + 1):
                attempts = attempt + 1
                try:
                    value = await asyncio.wait_for(scorer.score(metric, sample), settings.timeout_seconds)
                    value = float(value)
                    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ScoreError("score outside [0, 1] or not finite")
                    return Outcome(metric, sample.case_id, value, None, attempts)
                except MetricNotConfigured:
                    return Outcome(metric, sample.case_id, None, "not_configured", attempts)
                except ValueError as exc:  # invalid input: retrying cannot help
                    return Outcome(metric, sample.case_id, None, type(exc).__name__, attempts)
                except asyncio.TimeoutError:
                    error = "TimeoutError"
                except Exception as exc:  # transient judge failure: retry within the budget
                    error = type(exc).__name__
                if attempt < settings.retries:
                    await sleep(settings.backoff_seconds * (2 ** attempt))
        return Outcome(metric, sample.case_id, None, error, attempts)

    tasks = [one(s, m) for s in samples for m in metrics]
    return list(await asyncio.gather(*tasks))


def dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_record(settings: JudgeSettings, *, dataset: str | Path | None, pipeline: Mapping[str, Any],
               reference_source: str) -> dict[str, Any]:
    import platform

    versions = {"python": platform.python_version()}
    for pkg in ("ragas", "anthropic", "openai"):
        try:
            from importlib.metadata import version

            versions[pkg] = version(pkg)
        except Exception:
            versions[pkg] = "not installed"
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge": settings.public(),
        "pipeline": dict(pipeline),
        "dataset": {"path": str(dataset) if dataset else None,
                    "sha256": dataset_sha256(dataset) if dataset else None,
                    "reference_source": reference_source},
        "versions": versions,
        "note": "LLM judge output is not deterministic; compare runs by their records, not by exact scores.",
    }


def samples_from_rag(rag: Any, cases: Sequence[Mapping[str, Any]], tenant: str,
                     groups: frozenset[str] = frozenset()) -> tuple[list[Sample], str]:
    """Ask the pipeline each case and keep its answer with the raw parent passages it was built from.

    A case needs ``id``, ``question`` and ``reference``. Cases that only carry ``expect`` (a short expected answer) are
    accepted with ``reference_source = "expect"``, which the run record states, because a short key phrase is a weak
    reference for context recall.
    """
    sources = set()
    samples: list[Sample] = []
    for case in cases:
        if "reference" in case:
            reference, sources_key = case["reference"], "reference"
        else:
            reference, sources_key = case.get("expect", ""), "expect"
        sources.add(sources_key)
        if not reference:
            raise JudgeConfigError(f"case {case.get('id')!r} has no reference answer")
        a = rag.ask(case["question"], tenant=tenant, groups=groups)
        contexts = tuple(c.parent.text for c in a.sources)
        samples.append(Sample(case_id=str(case["id"]), question=case["question"], response=a.text,
                              contexts=contexts, reference=reference))
    if not sources:
        return samples, "none"
    return samples, ("reference" if sources == {"reference"} else "expect" if sources == {"expect"} else "mixed")


class RagasScorer:
    """Adapter from :class:`Scorer` to RAGAS' ``collections`` metrics. Imports ragas and the provider SDK lazily."""

    def __init__(self, settings: JudgeSettings):
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextRecall, Faithfulness

        self._metrics: dict[str, Any] = {}
        llm = llm_factory(settings.model, provider=settings.provider,
                          client=_async_client(settings.provider, settings.timeout_seconds))
        self._metrics["faithfulness"] = Faithfulness(llm=llm)
        self._metrics["context_recall"] = ContextRecall(llm=llm)
        if settings.embedding_provider:
            from ragas.embeddings import embedding_factory
            from ragas.metrics.collections import AnswerRelevancy

            emb = embedding_factory("openai", model=settings.embedding_model,
                                    client=_async_client("openai", settings.timeout_seconds))
            self._metrics["answer_relevancy"] = AnswerRelevancy(llm=llm, embeddings=emb)

    async def score(self, metric: str, sample: Sample) -> float:
        m = self._metrics.get(metric)
        if m is None:
            raise MetricNotConfigured(metric)
        if metric == "faithfulness":
            r = await m.ascore(user_input=sample.question, response=sample.response,
                               retrieved_contexts=list(sample.contexts))
        elif metric == "answer_relevancy":
            r = await m.ascore(user_input=sample.question, response=sample.response)
        else:
            r = await m.ascore(user_input=sample.question, retrieved_contexts=list(sample.contexts),
                               reference=sample.reference)
        return float(r.value)


def _async_client(provider: str, timeout_seconds: float) -> Any:
    # max_retries=0: retries belong to evaluate(), so one budget governs every call and the timeouts stay honest.
    if provider == "anthropic":
        import anthropic

        return anthropic.AsyncAnthropic(max_retries=0, timeout=timeout_seconds)
    import openai

    return openai.AsyncOpenAI(max_retries=0, timeout=timeout_seconds)


def run(samples: Sequence[Sample], scorer: Scorer, settings: JudgeSettings, *, dataset: str | Path | None,
        pipeline: Mapping[str, Any], reference_source: str, metrics: Sequence[str] = METRICS) -> EvalReport:
    """Synchronous entry point. Runs :func:`evaluate` and summarizes it with the run record."""
    started = time.monotonic()
    outcomes = asyncio.run(evaluate(samples, scorer, settings, metrics=metrics))
    record = run_record(settings, dataset=dataset, pipeline=pipeline, reference_source=reference_source)
    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return EvalReport(run=record, metrics=summarize(outcomes, metrics), outcomes=outcomes)


def write_report(report: EvalReport, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__: list[str] = ["METRICS", "JudgeSettings", "JudgeConfigError", "MetricNotConfigured", "ScoreError", "Sample",
                      "Outcome", "MetricSummary", "EvalReport", "evaluate", "summarize", "samples_from_rag",
                      "RagasScorer", "run", "write_report", "run_record", "dataset_sha256"]
