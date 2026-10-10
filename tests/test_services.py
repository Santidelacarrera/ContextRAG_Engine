"""Integration tests against real Redis / PostgreSQL processes, started on demand.
Skipped when the binaries are not installed."""
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from contextrag import Status, factory
from contextrag.core.cache import CachedEmbedder, RedisCache
from contextrag.core.embeddings import HashingEmbedder
from conftest import POLICY


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} never opened")


@pytest.fixture(scope="module")
def redis_url(tmp_path_factory):
    exe = shutil.which("redis-server")
    pytest.importorskip("redis")
    if not exe:
        pytest.skip("redis-server not installed")
    port = free_port()
    proc = subprocess.Popen([exe, "--port", str(port), "--save", "", "--bind", "127.0.0.1",
                             "--dir", str(tmp_path_factory.mktemp("redis"))], stdout=subprocess.DEVNULL)
    try:
        wait_port(port)
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        proc.terminate()
        proc.wait(10)


@pytest.fixture(scope="module")
def pg_dsn():
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


# ---- Redis -------------------------------------------------------------------------------

def test_redis_cache_roundtrip_ttl_and_incr(redis_url):
    c = RedisCache(redis_url, prefix="t1:")
    assert c.health() and c.get("k") is None
    c.set("k", "v", ttl=1)
    assert c.get("k") == "v"
    assert (c.incr("n"), c.incr("n")) == (1, 2)
    time.sleep(1.2)
    assert c.get("k") is None


def test_redis_backed_embedding_cache_survives_new_instances(redis_url):
    calls = []

    class Counting(HashingEmbedder):
        def embed(self, texts):
            calls.extend(texts)
            return super().embed(texts)

    CachedEmbedder(Counting(), RedisCache(redis_url, prefix="t2:"), "m").embed(["hola"])
    CachedEmbedder(Counting(), RedisCache(redis_url, prefix="t2:"), "m").embed(["hola"])  # e.g. another worker
    assert calls == ["hola"]


def test_pipeline_query_cache_on_redis(redis_url):
    rag = factory.offline()
    rag.cache = RedisCache(redis_url, prefix="t3:")
    rag.ingestor.cache = rag.cache
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    q = "¿Cuánto tiempo se conservan los registros de auditoría?"
    assert not rag.ask(q, tenant="acme").cached
    hit = rag.ask(q, tenant="acme")
    assert hit.cached and hit.status is Status.VERIFIED and hit.claims[0].evidence.text
    rag.ingestor.ingest_text(POLICY.replace("7 años", "8 años"), tenant="acme", doc_id="policy.md")
    after = rag.ask(q, tenant="acme")
    assert not after.cached and "8 años" in after.text


def test_redis_health_false_when_unreachable():
    pytest.importorskip("redis")
    assert RedisCache(f"redis://127.0.0.1:{free_port()}/0").health() is False


# ---- PostgreSQL --------------------------------------------------------------------------

def test_postgres_audit_persists_full_record(pg_dsn):
    from contextrag.audit import PostgresAudit

    audit = PostgresAudit(pg_dsn, dim=512)
    rag = factory.offline()
    rag.audit = audit
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    a = rag.ask("¿Cuánto tiempo se conservan los registros de auditoría?", tenant="acme", user="ana")
    rag.ask("¿Quién ganó el mundial de 1998?", tenant="acme")
    rag.ask("registros", tenant="other")
    assert audit.health()
    rows = audit.list("acme")
    assert [r["status"] for r in rows] == ["fallback", "verified"]
    ok = rows[1]
    assert ok["user"] == "ana" and ok["confidence"] == a.confidence
    assert ok["claims"][0]["evidence"]["text"] == a.claims[0].evidence.text  # jsonb round-trip
    assert ok["sources"][0]["doc_id"] == "policy.md"
    assert len(audit.list("other")) == 1 and len(audit.list("acme", limit=1)) == 1
    n = audit.conn.execute("SELECT count(*) FROM answer_audit WHERE query_embedding IS NOT NULL").fetchone()[0]
    assert n == 3  # embeddings stored (pgvector if present, float8[] otherwise)


def test_postgres_schema_is_idempotent_and_enforces_status_check(pg_dsn):
    import psycopg

    from contextrag.audit import PostgresAudit

    PostgresAudit(pg_dsn, dim=512)
    audit = PostgresAudit(pg_dsn, dim=512)
    with pytest.raises(psycopg.errors.CheckViolation):
        audit.conn.execute(
            "INSERT INTO answer_audit (id, tenant, question, status, answer, confidence, attempts, claims, sources) "
            "VALUES (gen_random_uuid(), 't', 'q', 'bogus', 'a', 0, 0, '[]', '[]')")
