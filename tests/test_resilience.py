"""Failure behaviour and concurrency: optional dependencies degrade, load is refused clearly, and requests never mix data."""
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import Status, factory
from contextrag.api.auth import KeyStore, new_entry
from contextrag.api.server import create_app
from contextrag.core.cache import CachedEmbedder, InMemoryCache, ResilientCache
from contextrag.core.config import Settings
from contextrag.core.embeddings import HashingEmbedder

FACT = "El plazo de prescripción de la acción es de cuatro años contados desde la notificación."
Q = "¿Cuál es el plazo de prescripción de la acción?"


class Switchable(InMemoryCache):
    """A cache that can be taken down and brought back, like a Redis that restarts."""

    def __init__(self):
        super().__init__()
        self.down = False

    def _check(self):
        if self.down:
            raise ConnectionError("redis unavailable at 10.0.0.9:6379")

    def get(self, key):
        self._check()
        return super().get(key)

    def set(self, key, value, ttl=None):
        self._check()
        super().set(key, value, ttl)

    def incr(self, key):
        self._check()
        return super().incr(key)

    def health(self):
        return not self.down


def rag_with(cache):
    r = factory.offline()
    r.cache = cache
    r.ingestor.cache = cache
    r.ingestor.ingest_text(FACT, tenant="t", doc_id="d.md", source="d.md")
    return r


# ---- an unavailable cache degrades to "no cache" ----------------------------------------------------------------------

def test_queries_embeddings_and_ingests_still_work_when_the_cache_is_down(caplog):
    backend = Switchable()
    cache = ResilientCache(backend)
    r = rag_with(cache)
    backend.down = True
    a = r.ask(Q, tenant="t")
    assert a.status is Status.VERIFIED and a.cached is False
    r.ingestor.ingest_text("Otro documento sobre plazos de caducidad.", tenant="t", doc_id="e.md", source="e.md")
    CachedEmbedder(HashingEmbedder(), cache, "m").embed(["hola"])
    assert cache.errors > 0 and cache.health() is False  # the outage is visible, not hidden
    assert f"contextrag_cache_errors_total {cache.errors}" in r.metrics_text()
    assert "10.0.0.9" not in caplog.text  # only the class of the failure is logged


def test_an_invalidation_lost_during_an_outage_is_applied_before_any_cached_answer_is_served():
    """The point of the wrapper. A document is replaced while the cache is down; when the cache returns, the answer cached
    before the change must not come back to life."""
    backend = Switchable()
    r = rag_with(ResilientCache(backend))
    first = r.ask(Q, tenant="t")
    assert first.status is Status.VERIFIED and "cuatro años" in first.text
    assert r.ask(Q, tenant="t").cached is True  # warm: the answer is cached

    backend.down = True
    r.ingestor.ingest_text("El plazo de prescripción de la acción es de dos años contados desde la notificación.",
                           tenant="t", doc_id="d.md", source="d.md")  # the invalidation fails here
    backend.down = False

    after = r.ask(Q, tenant="t")
    assert after.cached is False and "dos años" in after.text and "cuatro años" not in after.text
    assert r.ask(Q, tenant="t").cached is True  # and caching resumes normally once the invalidation was applied


def test_a_deleted_document_is_not_quoted_from_a_stale_cache_after_an_outage():
    backend = Switchable()
    r = rag_with(ResilientCache(backend))
    assert r.ask(Q, tenant="t").status is Status.VERIFIED  # cached
    backend.down = True
    r.ingestor.delete(tenant="t", doc_id="d.md")
    backend.down = False
    gone = r.ask(Q, tenant="t")
    assert gone.status is Status.FALLBACK and "cuatro años" not in gone.text


def test_reads_stay_disabled_while_an_invalidation_cannot_be_applied():
    backend = Switchable()
    cache = ResilientCache(backend)
    r = rag_with(cache)
    r.ask(Q, tenant="t")
    backend.down = True
    r.ingestor.ingest_text("Texto nuevo de otro tema distinto por completo.", tenant="t", doc_id="n.md", source="n.md")
    backend.down = False
    backend.set("probe", "x")
    assert cache._pending  # remembered
    assert cache.get("probe") == "x"  # reading applies the pending bump first, then proceeds
    assert not cache._pending


# ---- backpressure ------------------------------------------------------------------------------------------------------

class Blocking:
    def __init__(self):
        self.entered, self.release = threading.Event(), threading.Event()

    def generate(self, question, contexts, feedback=None, *, memory=None):
        self.entered.set()
        self.release.wait(10)
        return f"{FACT} [1]"


def test_requests_beyond_the_limit_are_refused_with_retry_after_instead_of_queuing_forever():
    llm = Blocking()
    cfg = replace(Settings(), max_concurrent_queries=1, busy_wait_seconds=0.05)
    r = factory.offline(cfg, llm=llm)
    r.ingestor.ingest_text(FACT, tenant="t", doc_id="d.md", source="d.md")
    client = TestClient(create_app(r))
    first: dict = {}
    t = threading.Thread(target=lambda: first.update(r=client.post("/query", headers={"X-Tenant": "t"}, json={"question": Q})))
    t.start()
    try:
        assert llm.entered.wait(5)
        busy = client.post("/query", headers={"X-Tenant": "t"}, json={"question": Q})
        assert busy.status_code == 503 and busy.headers["Retry-After"] == "2"
        assert "requests_rejected_busy_total" in r.metrics_text()
    finally:
        llm.release.set()
        t.join(10)
    assert first["r"].status_code == 200
    assert client.post("/query", headers={"X-Tenant": "t"}, json={"question": Q}).status_code == 200  # the slot was freed


def test_the_concurrency_limits_must_be_positive():
    with pytest.raises(ValueError):
        Settings(max_concurrent_queries=0)
    with pytest.raises(ValueError):
        Settings(busy_wait_seconds=-1)


# ---- isolation under concurrency -----------------------------------------------------------------------------------------

CASES = {"caso-a": "ALFA", "caso-b": "BRAVO", "caso-c": "CHARLIE", "caso-d": "DELTA"}


def multi_case_rag():
    r = factory.offline()
    for acl, name in CASES.items():
        r.ingestor.ingest_text(f"El expediente {name} fija la audiencia el día {10 + len(name)} de mayo en el Juzgado {name}.",
                               tenant="t", doc_id=f"{name.lower()}.md", source=f"{name.lower()}.md", acl=(acl,))
    return r


def test_concurrent_requests_for_different_cases_never_see_each_others_data():
    r = multi_case_rag()
    jobs = [(acl, name) for _ in range(60) for acl, name in CASES.items()]

    def work(job):
        acl, name = job
        a = r.ask("¿Cuándo es la audiencia del expediente?", tenant="t", groups=frozenset({acl}))
        others = [n for n in CASES.values() if n != name]
        blob = str(a.to_response())
        return name, a.status, [n for n in others if n in blob]

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, jobs))
    assert len(results) == 240
    assert all(leaked == [] for _, _, leaked in results), [x for x in results if x[2]][:3]
    assert all(status is Status.VERIFIED for _, status, _ in results)


def test_concurrent_api_requests_with_different_keys_stay_separated():
    entries, plain = {}, {}
    for kid, groups in (("ana", ("caso-a",)), ("beto", ("caso-b",)), ("carla", ("caso-c",))):
        key, digest, entry = new_entry(kid, "t", "reader", groups)
        entries[digest], plain[kid] = entry, key
    r = multi_case_rag()
    client = TestClient(create_app(r, keystore=KeyStore(entries)))
    expected = {"ana": "ALFA", "beto": "BRAVO", "carla": "CHARLIE"}

    def work(i):
        who = list(expected)[i % 3]
        resp = client.post("/query", headers={"X-API-Key": plain[who]}, json={"question": "¿Cuándo es la audiencia del expediente?"})
        text = resp.text
        return who, resp.status_code, [n for n in CASES.values() if n != expected[who] and n in text]

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(work, range(90)))
    assert all(code == 200 and leaked == [] for _, code, leaked in results), [x for x in results if x[2] or x[1] != 200][:3]


def test_concurrent_cached_answers_do_not_cross_cases():
    r = multi_case_rag()
    r.cache = InMemoryCache()
    r.ingestor.cache = r.cache
    barrier = threading.Barrier(8)

    def work(i):
        acl = list(CASES)[i % 4]
        barrier.wait(5)
        a = r.ask("¿Cuándo es la audiencia del expediente?", tenant="t", groups=frozenset({acl}))
        return CASES[acl], a.text

    with ThreadPoolExecutor(max_workers=8) as pool:
        out = list(pool.map(work, range(8)))
    for name, text in out:
        assert all(other not in text for other in CASES.values() if other != name)


def test_concurrent_ingest_and_queries_never_expose_a_half_replaced_document():
    r = factory.offline()
    r.ingestor.ingest_text("La versión uno del documento dice que el plazo es de tres meses.", tenant="t", doc_id="v.md", source="v.md")
    stop = threading.Event()
    bad: list[str] = []

    def writer():
        for i in range(40):
            r.ingestor.ingest_text(f"La versión {'dos' if i % 2 else 'uno'} del documento dice que el plazo es de "
                                   f"{'seis' if i % 2 else 'tres'} meses.", tenant="t", doc_id="v.md", source="v.md", force=True)
        stop.set()

    def reader():
        while not stop.is_set():
            a = r.ask("¿Qué dice la versión del documento sobre el plazo?", tenant="t")
            if a.status is Status.VERIFIED and not (("uno" in a.text and "tres" in a.text) or ("dos" in a.text and "seis" in a.text)):
                bad.append(a.text)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in threads]
    [t.join(60) for t in threads]
    assert bad == []  # an answer is never a mixture of two versions
