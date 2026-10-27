"""Stage tracing for the pipeline: durations and errors per stage, with no document, prompt or personal content.

Three layers, none mandatory:

* :class:`NullTracer`: the default for the interface. A span costs one attribute lookup.
* :class:`StageMetrics`: always wired by the pipeline. Counts calls, errors and milliseconds per stage name into the same
  counters ``/metrics`` already exposes, with the stage as the only label (a fixed list, so cardinality is bounded).
* :class:`OTelTracer`: optional export through OpenTelemetry (``pip install contextrag[otel]`` and
  ``CONTEXTRAG_TRACING=otel``). Asking for it without the package is an error, never a silent no-op, so nobody believes
  they are tracing when they are not. Configure the exporter the usual OpenTelemetry way (SDK and environment variables);
  this project does not start any collector.

What a span may carry. Attributes are numbers, booleans, or short identifier-like strings (lowercase letters, digits,
``_ : -``, at most 40 characters; no dots, so file names are never accepted). Anything else, including every free-text value, is dropped and counted. The tenant is
recorded only as a short hash. An exception is recorded by its class name; its message is never stored because messages
can contain document text, paths or credentials. This limits what can leak through tracing; it cannot stop a developer from
putting a document id in an identifier-like attribute, which is why call sites pass counts and states, not names.
"""
from __future__ import annotations

import contextvars
import hashlib
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

# One entry per stage. A name outside this list is a programming error, which keeps metric labels bounded.
STAGES = ("ask", "ingest", "chunk", "embed", "index", "retrieve", "rerank", "context", "entities", "conflicts",
          "generate", "verify", "verify.nli", "citations")
# No dots: a dot is how file names look (politica.md), and file names must not reach a span.
_SAFE_STRING = re.compile(r"^[a-z0-9_:\-]{1,40}$")

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("contextrag_request_id", default=None)


def short_hash(value: str) -> str:
    """A stable, non-reversible tag for an identifier such as a tenant (8 hex characters)."""
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def clean_attributes(attrs: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Keep numbers, booleans and identifier-like short strings. Returns (kept, number dropped)."""
    kept: dict[str, Any] = {}
    dropped = 0
    for key, value in attrs.items():
        if isinstance(value, (bool, int, float)) or (isinstance(value, str) and _SAFE_STRING.match(value)):
            kept[key] = value
        elif value is not None:
            dropped += 1
    return kept, dropped


class Tracer(Protocol):
    def span(self, name: str, **attrs: Any) -> AbstractContextManager[Any]: ...


class NullTracer:
    _NOOP = nullcontext()

    def span(self, name: str, **attrs: Any) -> AbstractContextManager[Any]:
        return self._NOOP


NULL_TRACER = NullTracer()


@dataclass
class SpanRecord:
    name: str
    parent: str | None
    duration_ms: float
    error: str | None
    attributes: dict[str, Any] = field(default_factory=dict)


class RecordingTracer:
    """Keeps finished spans in memory. For tests and local diagnosis; not an exporter."""

    def __init__(self, max_spans: int = 10_000):
        self.spans: list[SpanRecord] = []
        self.dropped_attributes = 0
        self._max, self._lock, self._local = max_spans, threading.Lock(), threading.local()

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        stack: list[str] = self._local.__dict__.setdefault("stack", [])
        kept, dropped = clean_attributes(attrs)
        parent = stack[-1] if stack else None
        stack.append(name)
        start, error = time.monotonic(), None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            stack.pop()
            with self._lock:
                self.dropped_attributes += dropped
                if len(self.spans) < self._max:
                    self.spans.append(SpanRecord(name, parent, (time.monotonic() - start) * 1000, error, kept))

    def names(self) -> list[str]:
        return [s.name for s in self.spans]


class StageMetrics:
    """Wraps another tracer (or none) and feeds per-stage counters. ``count`` is ``ContextRAG.count``."""

    def __init__(self, count: Callable[..., None], inner: Tracer | None = None):
        self._count, self._inner = count, inner or NULL_TRACER

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}; add it to STAGES (stage names are metric labels)")
        start, error = time.monotonic(), None
        try:
            with self._inner.span(name, **attrs):
                yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            ms = (time.monotonic() - start) * 1000
            self._count(f'stage_calls_total{{stage="{name}"}}')
            self._count(f'stage_duration_ms_sum{{stage="{name}"}}', ms)
            if error is not None:
                self._count(f'stage_errors_total{{stage="{name}"}}')


class OTelTracer:
    """OpenTelemetry export. Requires ``opentelemetry-api``; the SDK and exporter are the deployment's choice."""

    def __init__(self, tracer: Any = None):
        if tracer is None:
            try:
                from opentelemetry import trace
            except ImportError:
                raise RuntimeError("tracing was requested (CONTEXTRAG_TRACING=otel) but opentelemetry is not installed; "
                                   "install contextrag[otel] or unset CONTEXTRAG_TRACING") from None
            tracer = trace.get_tracer("contextrag")
        self._tracer = tracer
        self.dropped_attributes = 0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[None]:
        kept, dropped = clean_attributes(attrs)
        self.dropped_attributes += dropped
        rid = request_id_var.get()
        if rid and _SAFE_STRING.match(rid.lower()):
            kept["request_id"] = rid.lower()
        # record_exception=False: an exception message can hold document text, paths or credentials
        with self._tracer.start_as_current_span(name, attributes=kept, record_exception=False,
                                                set_status_on_exception=False) as otel_span:
            try:
                yield
            except BaseException as exc:
                otel_span.set_attribute("error.class", type(exc).__name__)
                otel_span.set_attribute("error", True)
                raise


def tracer_from_env(env: dict[str, str] | None = None) -> Tracer | None:
    """The optional exporting tracer selected by ``CONTEXTRAG_TRACING`` (``off`` by default)."""
    import os

    mode = (os.environ if env is None else env).get("CONTEXTRAG_TRACING", "off").strip().lower()
    if mode in ("", "off", "0", "false", "no"):
        return None
    if mode == "otel":
        return OTelTracer()
    raise ValueError(f"CONTEXTRAG_TRACING must be 'off' or 'otel', got {mode!r}")
