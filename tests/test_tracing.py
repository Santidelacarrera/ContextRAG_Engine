"""Stage tracing: every stage is measured, errors are recorded by class, and no content reaches a span or a counter."""
import pytest

from contextrag import Status, factory
from contextrag.core.tracing import (STAGES, RecordingTracer, clean_attributes, short_hash, tracer_from_env)

DOC = "El plazo de prescripción de la acción es de cuatro años contados desde la notificación."
Q = "¿Cuál es el plazo de prescripción de la acción?"
SECRET = "RUT 12.345.678-9 caso ORCHID-7731 confidencial"


def traced(**kw):
    rec = RecordingTracer()
    r = factory.offline(tracer=rec, **kw)
    r.ingestor.ingest_text(DOC + " " + SECRET, tenant="acme", doc_id="d.md", source="d.md")
    return r, rec


def test_a_question_produces_the_full_stage_chain_under_one_root():
    r, rec = traced()
    a = r.ask(Q, tenant="acme")
    assert a.status is Status.VERIFIED
    names = rec.names()
    for stage in ("ask", "context", "generate", "verify", "verify.nli", "citations", "retrieve", "rerank"):
        assert stage in names, (stage, names)
    roots = [s for s in rec.spans if s.name == "ask"]
    assert len(roots) == 1 and roots[0].parent is None
    assert {s.parent for s in rec.spans if s.name in ("retrieve", "context", "generate")} <= {"ask", "retrieve", "context"}


def test_ingestion_stages_are_chunk_embed_and_index():
    _, rec = traced()
    assert {"ingest", "chunk", "embed", "index"} <= set(rec.names())


def test_durations_are_non_negative_and_nested_spans_do_not_exceed_their_parent():
    r, rec = traced()
    r.ask(Q, tenant="acme")
    root = next(s for s in rec.spans if s.name == "ask")
    assert root.duration_ms >= 0
    children = [s for s in rec.spans if s.parent == "ask"]
    assert all(c.duration_ms <= root.duration_ms + 1e-6 for c in children)


def test_metrics_count_calls_and_milliseconds_per_stage_and_stay_bounded():
    r, _ = traced()
    r.ask(Q, tenant="acme")
    text = r.metrics_text()
    assert 'contextrag_stage_calls_total{stage="retrieve"} ' in text
    assert 'contextrag_stage_duration_ms_sum{stage="verify"} ' in text
    labels = {line.split('stage="')[1].split('"')[0] for line in text.splitlines() if 'stage="' in line}
    assert labels <= set(STAGES)  # a bounded, documented set of label values


def test_an_unknown_stage_is_a_programming_error_not_a_new_label():
    r = factory.offline()
    with pytest.raises(ValueError, match="unknown stage"):
        with r.tracer.span("made_up_stage"):
            pass


def test_a_failing_stage_records_its_class_and_the_error_propagates():
    class Boom:
        def generate(self, *a, **k):
            raise KeyError("secret-path /home/deploy/key.pem")

    r = factory.offline(llm=Boom(), tracer=RecordingTracer())
    r.ingestor.ingest_text(DOC, tenant="acme", doc_id="d.md", source="d.md")
    from contextrag.generation.llm import GenerationError

    with pytest.raises(GenerationError) as info:
        r.ask(Q, tenant="acme")
    assert str(info.value) == "KeyError" and isinstance(info.value.__cause__, KeyError)  # class only; cause kept internally
    gen = next(s for s in r.tracer._inner.spans if s.name == "generate")
    assert gen.error == "KeyError"
    assert 'contextrag_stage_errors_total{stage="generate"} 1' in r.metrics_text()
    assert "/home/deploy" not in str(r.tracer._inner.spans) and "/home/deploy" not in r.metrics_text()


def test_span_attributes_carry_no_document_question_or_tenant_name():
    r, rec = traced()
    r.ask(Q, tenant="acme")
    blob = repr([(s.name, s.attributes) for s in rec.spans])
    for leak in ("acme", "ORCHID", "12.345.678", "prescripción", "cuatro años", "/home"):
        assert leak not in blob, leak
    root = next(s for s in rec.spans if s.name == "ask")
    assert root.attributes["tenant"] == short_hash("acme") and len(root.attributes["tenant"]) == 8


def test_attribute_cleaning_keeps_identifiers_and_numbers_and_drops_free_text():
    kept, dropped = clean_attributes({"stage": "verify", "n": 3, "ok": True, "free": "texto libre con espacios",
                                      "doc": "politica.md", "none": None, "long": "x" * 41})
    assert kept == {"stage": "verify", "n": 3, "ok": True}
    assert dropped == 3  # free text, a document name, an over-long value; None is simply absent


def test_the_recording_tracer_is_bounded():
    rec = RecordingTracer(max_spans=2)
    for _ in range(5):
        with rec.span("ingest"):
            pass
    assert len(rec.spans) == 2


def test_tracing_is_off_by_default_and_refuses_unknown_modes(monkeypatch):
    monkeypatch.delenv("CONTEXTRAG_TRACING", raising=False)
    assert tracer_from_env({}) is None
    with pytest.raises(ValueError):
        tracer_from_env({"CONTEXTRAG_TRACING": "datadog"})


# ---- OpenTelemetry, against the real SDK -------------------------------------------------------------------

def otel_pair():
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider.get_tracer("test")


def test_the_otel_tracer_exports_the_stages_with_the_safe_attributes_only():
    from contextrag.core.tracing import OTelTracer, request_id_var

    exporter, tracer = otel_pair()
    r = factory.offline(tracer=OTelTracer(tracer))
    r.ingestor.ingest_text(DOC + " " + SECRET, tenant="acme", doc_id="d.md", source="d.md")
    request_id_var.set("req-abc123")
    try:
        r.ask(Q, tenant="acme")
    finally:
        request_id_var.set(None)
    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert {"ingest", "chunk", "embed", "index", "ask", "retrieve", "generate", "verify"} <= names
    dumped = repr([(s.name, dict(s.attributes)) for s in spans])
    for leak in ("acme", "ORCHID", "12.345.678", "prescripción", "/home"):
        assert leak not in dumped, leak
    ask = next(s for s in spans if s.name == "ask")
    assert ask.attributes["request_id"] == "req-abc123"
    assert ask.attributes["tenant"] == short_hash("acme")


def test_an_otel_error_records_the_class_and_the_status_without_the_message():
    from contextrag.core.tracing import OTelTracer

    exporter, tracer = otel_pair()
    t = OTelTracer(tracer)
    with pytest.raises(ValueError):
        with t.span("retrieve"):
            raise ValueError("ruta /etc/secreto y RUT 1.2.3-4")
    span = exporter.get_finished_spans()[0]
    assert span.attributes["error.class"] == "ValueError" and span.attributes["error"] is True
    assert "secreto" not in repr(span.events) and "RUT" not in repr(span.attributes)  # no exception event with its text
    assert not span.events


def test_asking_for_otel_without_the_package_is_an_error_not_a_silent_no_op(monkeypatch):
    import builtins

    from contextrag.core.tracing import OTelTracer

    real = builtins.__import__

    def no_otel(name, *a, **k):
        if name.startswith("opentelemetry"):
            raise ImportError("simulated")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_otel)
    with pytest.raises(RuntimeError, match="opentelemetry is not installed"):
        OTelTracer()


def test_the_otel_mode_is_selected_by_environment_only():
    from contextrag.core.tracing import OTelTracer

    assert isinstance(tracer_from_env({"CONTEXTRAG_TRACING": "otel"}), OTelTracer)
