import base64

import pytest

from contextrag import Status, factory
from contextrag.core.ingestion import EmptyDocumentError, from_text, html_to_text, load_bytes
from contextrag.generation.llm import GenerationError
from conftest import make_pdf

HTML = """<!doctype html><html><head><title>Ignored title</title><style>p{color:red}</style>
<script>var secret = "no debe indexarse";</script></head>
<body><h1>Política &amp; retención</h1>
<p>Los registros de auditoría se conservan durante <b>7</b> años.</p>
<h2>Cifrado</h2><ul><li>AES-256 en reposo</li><li>Claves rotadas cada 12&nbsp;meses</li></ul>
<noscript>activa JavaScript</noscript></body></html>"""


# ---- HTML ----------------------------------------------------------------------------------

def test_html_to_text_keeps_visible_text_and_headings_only():
    t = html_to_text(HTML)
    assert t.startswith("# Política & retención")
    assert "## Cifrado" in t and "Los registros de auditoría se conservan durante 7 años." in t.replace("\n", " ").replace("  ", " ")
    assert "secret" not in t and "color:red" not in t and "Ignored title" not in t and "JavaScript" not in t
    assert "AES-256 en reposo" in t and "<" not in t


def test_html_documents_are_answerable_with_sections_and_exact_offsets():
    doc = from_text(HTML, "politica.html")
    assert doc.kind == "markdown"
    r = factory.offline()
    r.ingestor.ingest(doc, tenant="t", doc_id="politica.html", source="politica.html")
    a = r.ask("¿Cuánto tiempo se conservan los registros de auditoría?", tenant="t")
    assert a.status is Status.VERIFIED and a.sources[0].parent.section == "Política & retención"
    ev = a.claims[0].evidence
    assert doc.text[ev.start : ev.end] == ev.text  # coordinates refer to the extracted text


def test_html_via_ingest_path_and_api(tmp_path):
    (tmp_path / "a.html").write_text(HTML, encoding="utf-8")
    r = factory.offline()
    [done] = r.ingestor.ingest_path(tmp_path, tenant="t")
    assert done["doc_id"] == "a.html" and done["kind"] == "markdown" and done["children"] >= 1


# ---- empty documents -----------------------------------------------------------------------

def blank_pdf():
    return make_pdf([""])


def test_scanned_or_empty_documents_are_refused_not_silently_indexed():
    r = factory.offline()
    with pytest.raises(EmptyDocumentError):
        r.ingestor.ingest(load_bytes(blank_pdf(), "scan.pdf"), tenant="t", doc_id="scan.pdf")
    for empty in ("", "   \n\n  ", "<html><script>x</script></html>"):
        with pytest.raises(EmptyDocumentError):
            r.ingestor.ingest_text(empty, tenant="t", doc_id="e.html")


def test_a_failed_reingest_leaves_the_previous_version_intact():
    r = factory.offline()
    r.ingestor.ingest_text("Los registros se conservan 7 años.", tenant="t", doc_id="d.md")
    with pytest.raises(EmptyDocumentError):
        r.ingestor.ingest_text("  ", tenant="t", doc_id="d.md")
    assert r.ask("¿cuánto se conservan los registros?", tenant="t").status is Status.VERIFIED


def test_ingest_path_reports_skipped_documents(tmp_path):
    (tmp_path / "ok.md").write_text("Los registros se conservan 7 años.")
    (tmp_path / "blank.txt").write_text("   ")
    done = {d["doc_id"]: d for d in factory.offline().ingestor.ingest_path(tmp_path, tenant="t")}
    assert "children" in done["ok.md"] and "no extractable text" in done["blank.txt"]["skipped"]


def test_api_returns_422_for_empty_documents_and_serves_html():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from contextrag.api.server import create_app

    c = TestClient(create_app(factory.offline()))
    h = {"X-Tenant": "t"}
    pdf = base64.b64encode(blank_pdf()).decode()
    r = c.post("/ingest", json={"documents": [{"doc_id": "scan.pdf", "content_base64": pdf}]}, headers=h)
    assert r.status_code == 422 and "no extractable text" in r.json()["detail"]
    assert c.post("/ingest", json={"documents": [{"doc_id": "e.md", "text": "   "}]}, headers=h).status_code == 422
    ok = c.post("/ingest", json={"documents": [{"doc_id": "p.html", "text": HTML}]}, headers=h)
    assert ok.status_code == 200 and ok.json()["documents"][0]["kind"] == "markdown"
    j = c.post("/query", json={"question": "¿Cuánto tiempo se conservan los registros de auditoría?"}, headers=h).json()
    assert j["status"] == "verified"


# ---- generation outages --------------------------------------------------------------------

class Down:
    def generate(self, *a, **k):
        raise ConnectionError("api.anthropic.com unreachable")


def test_llm_failures_surface_as_generation_error_not_as_a_misleading_fallback():
    r = factory.offline(llm=Down())
    r.ingestor.ingest_text("Los registros se conservan 7 años.", tenant="t", doc_id="d")
    with pytest.raises(GenerationError, match="ConnectionError"):
        r.ask("¿cuánto se conservan los registros?", tenant="t")
    assert r.audit.records == []  # nothing was answered, so nothing is audited as an answer
    assert 'status="' not in r.metrics_text()


def test_generation_error_is_not_cached_and_recovers_when_the_llm_returns():
    class Flaky:
        calls = 0

        def generate(self, question, contexts, feedback=None):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise GenerationError("quota")
            return "Los registros se conservan 7 años. [1]"

    r = factory.offline(llm=Flaky())
    r.ingestor.ingest_text("Los registros se conservan 7 años.", tenant="t", doc_id="d")
    with pytest.raises(GenerationError):
        r.ask("¿cuánto se conservan los registros?", tenant="t")
    assert r.ask("¿cuánto se conservan los registros?", tenant="t").status is Status.VERIFIED


def test_api_maps_generation_failure_to_503_with_retry_after():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from contextrag.api.server import create_app

    rag = factory.offline(llm=Down())
    rag.ingestor.ingest_text("Los registros se conservan 7 años.", tenant="t", doc_id="d")
    c = TestClient(create_app(rag))
    r = c.post("/query", json={"question": "¿cuánto se conservan los registros?"}, headers={"X-Tenant": "t"})
    assert r.status_code == 503 and r.json()["error"] == "generation_unavailable" and r.headers["Retry-After"] == "5"
    assert "unreachable" not in r.text  # internal error details are logged, not leaked to the client
    assert c.get("/healthz").status_code == 200  # the service itself is still up


def test_a_misbehaving_reranker_fails_loudly_instead_of_skipping_the_relevance_check():
    class Truncating:
        min_score = 0.0

        def score(self, query, passages):
            return [1.0] * max(0, len(passages) - 1)  # one score short

    r = factory.offline()
    r.ingestor.ingest_text("Los registros se conservan 7 años. Los gatos duermen.", tenant="t", doc_id="d")
    r.retriever.reranker = factory.LexicalReranker()  # retrieval fine; only the answer-evidence gate is broken
    r.reranker = Truncating()
    with pytest.raises(ValueError):
        r.ask("¿cuánto se conservan los registros?", tenant="t")
