"""Real Redis, real outage: the server is stopped with its data saved, a document changes while it is down, and the server
comes back with the *old* cached answer still in its snapshot. The answer must not come back.

Uses a redis-server process started by the test (skipped when the binary is absent), like tests/test_services.py.
"""
import shutil
import socket
import subprocess
import time

import pytest

pytest.importorskip("redis")
import redis as redis_lib  # noqa: E402

from contextrag import Status, factory  # noqa: E402
from contextrag.core.cache import RedisCache, ResilientCache  # noqa: E402

FACT_OLD = "El plazo de prescripción de la acción es de cuatro años contados desde la notificación."
FACT_NEW = "El plazo de prescripción de la acción es de dos años contados desde la notificación."
Q = "¿Cuál es el plazo de prescripción de la acción?"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Redis:
    def __init__(self, exe, port, dump_dir):
        self.exe, self.port, self.dir = exe, port, dump_dir
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([self.exe, "--port", str(self.port), "--bind", "127.0.0.1", "--save", "",
                                      "--dir", str(self.dir), "--dbfilename", "snap.rdb"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                redis_lib.Redis(port=self.port).ping()
                return self
            except redis_lib.ConnectionError:
                time.sleep(0.05)
        raise RuntimeError("redis did not start")

    def save_and_stop(self):
        try:
            redis_lib.Redis(port=self.port).execute_command("SHUTDOWN", "SAVE")  # graceful: snapshot first
        except redis_lib.ConnectionError:
            pass  # expected: the server closes the connection as it shuts down
        self.proc.wait(10)
        assert self.proc.returncode is not None

    def url(self):
        return f"redis://127.0.0.1:{self.port}/0"


@pytest.fixture
def redis_server(tmp_path):
    exe = shutil.which("redis-server")
    if not exe:
        pytest.skip("redis-server not installed")
    server = Redis(exe, free_port(), tmp_path).start()
    yield server
    if server.proc and server.proc.poll() is None:
        server.proc.kill()


def test_a_document_changed_during_a_redis_outage_is_not_answered_from_the_snapshot(redis_server, tmp_path):
    cache = ResilientCache(RedisCache(redis_server.url(), prefix="outage:"))
    rag = factory.offline()
    rag.cache = cache
    rag.ingestor.cache = cache
    rag.ingestor.ingest_text(FACT_OLD, tenant="t", doc_id="d.md", source="d.md")
    warm = rag.ask(Q, tenant="t")
    assert warm.status is Status.VERIFIED and "cuatro años" in warm.text
    assert rag.ask(Q, tenant="t").cached is True  # the answer is now in Redis

    redis_server.save_and_stop()  # the snapshot with the old answer is on disk
    rag.ingestor.ingest_text(FACT_NEW, tenant="t", doc_id="d.md", source="d.md")  # invalidation fails: Redis is down
    assert cache.errors > 0

    redis_server.start()  # the server comes back and reloads the snapshot, old answer included
    after = rag.ask(Q, tenant="t")
    assert after.cached is False
    assert "dos años" in after.text and "cuatro años" not in after.text


def test_the_service_keeps_answering_while_redis_is_down(redis_server):
    cache = ResilientCache(RedisCache(redis_server.url(), prefix="down:"))
    rag = factory.offline()
    rag.cache = cache
    rag.ingestor.cache = cache
    rag.ingestor.ingest_text(FACT_OLD, tenant="t", doc_id="d.md", source="d.md")
    redis_server.save_and_stop()
    a = rag.ask(Q, tenant="t")
    assert a.status is Status.VERIFIED and cache.health() is False
    assert f"contextrag_cache_errors_total {cache.errors}" in rag.metrics_text()
    redis_server.start()
    assert rag.ask(Q, tenant="t").status is Status.VERIFIED
