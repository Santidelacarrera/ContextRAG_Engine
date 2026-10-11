import os
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path

import pytest

from contextrag import factory


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def pg_dsn():
    """A throwaway PostgreSQL 16 server (superuser DSN). Skipped when the binaries are not installed."""
    pytest.importorskip("psycopg")
    bindir = next(iter(sorted(Path("/usr/lib/postgresql").glob("*/bin"))), None)
    if not bindir or not (bindir / "initdb").exists():
        pytest.skip("postgresql not installed")
    data = Path(tempfile.mkdtemp(prefix="contextrag-pg-"))  # must be reachable by the postgres OS user
    port = free_port()
    prefix = []
    if os.geteuid() == 0:  # postgres refuses to run as root
        if not shutil.which("runuser"):
            pytest.skip("cannot drop root privileges")
        os.chmod(data, 0o700)
        shutil.chown(data, "postgres")
        prefix = ["runuser", "-u", "postgres", "--"]
    run = lambda *a, **k: subprocess.run([*prefix, *a], check=True, capture_output=True, **k)
    run(str(bindir / "initdb"), "-D", str(data), "-A", "trust", "-U", "postgres")
    run(str(bindir / "pg_ctl"), "-D", str(data), "-l", str(data / "server.log"),
        "-o", f"-p {port} -k {data} -c listen_addresses=127.0.0.1", "-w", "-t", "30", "start")
    try:
        yield f"postgresql://postgres@127.0.0.1:{port}/postgres"
    finally:
        run(str(bindir / "pg_ctl"), "-D", str(data), "-m", "immediate", "stop")
        shutil.rmtree(data, ignore_errors=True)


@pytest.fixture(scope="module")
def pg_app_dsn(pg_dsn):
    """Connection string of a plain login role: no superuser, no BYPASSRLS, not the table owner."""
    import psycopg

    with psycopg.connect(pg_dsn, autocommit=True) as c:
        c.execute("DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'contextrag_app') "
                  "THEN CREATE ROLE contextrag_app LOGIN; END IF; END $$")
    return pg_dsn.replace("postgresql://postgres@", "postgresql://contextrag_app@", 1)

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
