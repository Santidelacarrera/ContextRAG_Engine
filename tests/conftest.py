import pytest

from contextrag import factory

POLICY = """# Retención de datos

La política de retención establece que los registros de auditoría se conservan durante 7 años. Los registros de acceso se eliminan a los 90 días.

# Cifrado

Todos los datos en reposo se cifran con AES-256. Las claves se rotan cada 12 meses.
"""


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal valid multi-page PDF containing one line of text per page."""
    objs: list[bytes] = []
    n = len(pages)
    page_ids = [3 + 2 * i for i in range(n)]
    font_id = 3 + 2 * n
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in page_ids) + b"] /Count %d >>" % n)
    for i, text in enumerate(pages):
        stream = b"BT /F1 12 Tf 50 700 Td (" + text.encode().replace(b"(", b"\\(").replace(b")", b"\\)") + b") Tj ET"
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
                    b"/Resources << /Font << /F1 %d 0 R >> >> >>" % (page_ids[i] + 1, font_id))
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.fixture
def rag():
    r = factory.offline()
    r.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    return r
