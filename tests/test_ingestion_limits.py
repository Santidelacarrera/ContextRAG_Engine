"""Upload validation: files are refused before parsing, with stable codes and without echoing their content."""
import base64
import io

import pytest
from fastapi.testclient import TestClient

from conftest import make_pdf
from contextrag import factory
from contextrag.api.server import create_app
from contextrag.core.config import Settings
from contextrag.core.ingestion import DocumentRejected, load_bytes

SECRET = "ORCHID-7731-confidential"


def reject(data, name, settings=None):
    with pytest.raises(DocumentRejected) as info:
        load_bytes(data, name, settings)
    assert SECRET not in str(info.value)  # a refusal never echoes the file
    return info.value.code


def pdf_with_pages(n, **encrypt):
    from pypdf import PdfWriter

    w = PdfWriter()
    for _ in range(n):
        w.add_blank_page(width=200, height=200)
    if encrypt:
        w.encrypt(**encrypt)
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


# ---- size and signature --------------------------------------------------------------------

def test_oversize_files_are_refused_before_parsing():
    assert reject(b"x" * 101, "big.txt", Settings(max_document_bytes=100)) == "too_large"
    assert load_bytes(b"x" * 100, "ok.txt", Settings(max_document_bytes=100)).text == "x" * 100  # the limit is inclusive


def test_a_file_named_pdf_must_carry_a_pdf_signature():
    assert reject(f"just text {SECRET}".encode(), "report.pdf") == "signature_mismatch"


@pytest.mark.parametrize("magic", [b"%PDF-1.7\n", b"PK\x03\x04", b"\x89PNG\r\n", b"\x7fELF", b"\xff\xd8\xff"])
def test_a_text_file_with_a_binary_signature_is_refused(magic):
    assert reject(magic + SECRET.encode(), "notes.txt") == "signature_mismatch"


def test_nul_bytes_in_a_text_file_mean_binary_content():
    assert reject(b"hello\x00" + SECRET.encode(), "notes.md") == "binary_content"


def test_plain_text_markdown_and_code_still_load():
    for name, body in (("a.txt", "texto"), ("b.md", "# t\n\ncuerpo"), ("c.py", "def f():\n    return 1\n")):
        assert load_bytes(body.encode(), name).text == body


# ---- PDFs ---------------------------------------------------------------------------------

def test_a_password_protected_pdf_is_refused_as_encrypted():
    assert reject(pdf_with_pages(1, user_password="secret", owner_password="owner"), "locked.pdf") == "encrypted"


def test_a_pdf_with_only_an_owner_password_opens_with_the_empty_user_password():
    data = pdf_with_pages(1, user_password="", owner_password="owner")
    assert load_bytes(data, "restricted.pdf").kind == "pdf"  # many "protected" PDFs are readable; they are not refused


def test_too_many_pages_are_refused_without_extracting_them():
    assert reject(pdf_with_pages(5), "long.pdf", Settings(max_pdf_pages=3)) == "too_many_pages"
    assert load_bytes(pdf_with_pages(3), "ok.pdf", Settings(max_pdf_pages=3)).kind == "pdf"


def test_corrupt_pdfs_are_malformed_and_the_parser_message_is_not_passed_on():
    err = pytest.raises(DocumentRejected, load_bytes, b"%PDF-1.4\n" + SECRET.encode() * 3, "broken.pdf").value
    assert err.code == "malformed_pdf" and SECRET not in str(err)


def test_a_valid_pdf_still_loads_with_page_offsets():
    doc = load_bytes(make_pdf(["First page about apples.", "Second page about oranges."]), "f.pdf")
    assert len(doc.page_starts) == 2


# ---- configuration ------------------------------------------------------------------------

def test_limits_must_be_positive_and_can_come_from_the_environment(monkeypatch):
    with pytest.raises(ValueError, match="must be >= 1"):
        Settings(max_document_bytes=0)
    with pytest.raises(ValueError, match="must be >= 1"):
        Settings(max_pdf_pages=0)
    monkeypatch.setenv("CONTEXTRAG_MAX_PDF_PAGES", "7")
    monkeypatch.setenv("CONTEXTRAG_MAX_DOCUMENT_BYTES", "1234")
    cfg = Settings.from_env()
    assert (cfg.max_pdf_pages, cfg.max_document_bytes) == (7, 1234)


# ---- API mapping --------------------------------------------------------------------------

def ingest(client, name, data):
    return client.post("/ingest", headers={"X-Tenant": "acme"}, json={"documents": [
        {"doc_id": name, "content_base64": base64.b64encode(data).decode(), "source": name}]})


def test_the_api_maps_oversize_to_413_and_other_refusals_to_422_with_the_code():
    client = TestClient(create_app(factory.offline(Settings(max_document_bytes=200, max_pdf_pages=2))))
    r = ingest(client, "big.txt", b"x" * 500)
    assert r.status_code == 413 and "too_large" in r.json()["detail"]
    r = ingest(client, "fake.pdf", b"plain text")
    assert r.status_code == 422 and "signature_mismatch" in r.json()["detail"]
    r = ingest(client, "img.txt", b"\x89PNG....")
    assert r.status_code == 422 and "signature_mismatch" in r.json()["detail"]
    roomy = TestClient(create_app(factory.offline()))  # the locked PDF is bigger than the tiny limit above
    r = ingest(roomy, "locked.pdf", pdf_with_pages(1, user_password="s", owner_password="o"))
    assert r.status_code == 422 and "encrypted" in r.json()["detail"]


def test_a_refused_upload_leaves_the_indexed_version_untouched():
    rag = factory.offline()
    client = TestClient(create_app(rag))
    ok = ingest(client, "policy.txt", "Los registros de auditoría se conservan siete años completos.".encode())
    assert ok.status_code == 200
    before = rag.index.list_documents("acme")
    assert ingest(client, "policy.txt", b"\x89PNG corrupted replacement").status_code == 422
    assert rag.index.list_documents("acme") == before
