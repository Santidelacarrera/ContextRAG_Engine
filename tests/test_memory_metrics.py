"""Compaction worker metrics: names, semantics, retries, cancellation and recovery, and exposure through /metrics."""
import threading

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from contextrag.api.server import create_app
from contextrag.core.security import TenantCipher
from contextrag.memory.metrics import MemoryMetrics, default_metrics
from contextrag.memory.store import ConversationStore
from contextrag.memory.summarizer import ExtractiveSummarizer
from contextrag.memory.worker import CompactionWorker
from contextrag.factory import offline

KEY = b"0123456789abcdef0123456789abcdef"
DEPTH = "contextrag_memory_queue_depth"
DURATION_COUNT = "contextrag_memory_compaction_duration_seconds_count"
SERVED = "contextrag_memory_compaction_jobs_served_total"
FAILED = "contextrag_memory_compaction_jobs_failed_total"
RETRIES = "contextrag_memory_compaction_retries_total"
CANCELLED = "contextrag_memory_compaction_cancelled_total"


def value(metrics, name):
    return metrics.registry.get_sample_value(name) or 0.0


def store_with(conversations=1, turns=4, max_attempts=3, clock_t=1000.0):
    s = ConversationStore(":memory:", cipher=TenantCipher(KEY), max_attempts=max_attempts, clock=lambda: clock_t)
    for c in range(conversations):
        cid = f"c{c}"
        s.create_conversation("acme", cid)
        for i in range(turns):
            s.append_turn("acme", cid, "user" if i % 2 == 0 else "assistant", f"decision number {c}-{i}")
    return s


def schedule(worker, cid):
    job = worker.schedule("acme", cid, threshold_tokens=0, keep_recent_turns=1)
    assert job is not None
    return job


class Blocking(ExtractiveSummarizer):
    """Holds every attempt until released, so a test can observe a job that is running."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def summarize(self, turns):
        self.entered.set()
        self.release.wait(10)
        return super().summarize(turns)


class AlwaysFails(ExtractiveSummarizer):
    def summarize(self, turns):
        raise ValueError("model unavailable")


# ---- semantics ---------------------------------------------------------------------------------------

def test_a_successful_job_is_served_timed_and_leaves_the_depth_at_zero():
    s = store_with()
    m = MemoryMetrics(CollectorRegistry())
    w = CompactionWorker(s, ExtractiveSummarizer(), workers=1, metrics=m)
    try:
        schedule(w, "c0")
        assert w.drain(10)
    finally:
        w.close()
    assert value(m, SERVED) == 1
    assert value(m, FAILED) == 0 and value(m, RETRIES) == 0
    assert value(m, DURATION_COUNT) == 1
    assert value(m, DEPTH) == 0


def test_a_terminal_failure_counts_once_as_failed_and_each_requeue_as_a_retry():
    s = store_with(max_attempts=2)
    m = MemoryMetrics(CollectorRegistry())
    w = CompactionWorker(s, AlwaysFails(), workers=1, metrics=m)
    try:
        job = schedule(w, "c0")
        assert w.drain(10)
    finally:
        w.close()
    assert s.job_state(job) == "failed"
    assert value(m, FAILED) == 1  # the job is terminal once
    assert value(m, RETRIES) == 1  # attempt 1 was requeued, attempt 2 was terminal
    assert value(m, DURATION_COUNT) == 2  # every attempt is timed, retries included
    assert value(m, SERVED) == 0
    assert value(m, DEPTH) == 0


def test_queue_depth_reports_a_running_job_and_the_gauge_is_store_truth():
    s = store_with()
    m = MemoryMetrics(CollectorRegistry())
    blocking = Blocking()
    w = CompactionWorker(s, blocking, workers=1, metrics=m)
    try:
        schedule(w, "c0")
        assert blocking.entered.wait(10)
        assert value(m, DEPTH) == 1  # running counts as pending work
        blocking.release.set()
        assert w.drain(10)
    finally:
        blocking.release.set()
        w.close()
    assert value(m, DEPTH) == 0


def test_shutdown_cancels_unstarted_work_leaves_it_queued_and_recovery_resumes_it():
    s = store_with(conversations=2)
    m = MemoryMetrics(CollectorRegistry())
    blocking = Blocking()
    w = CompactionWorker(s, blocking, workers=1, metrics=m)
    queued = None
    try:
        schedule(w, "c0")
        assert blocking.entered.wait(10)
        queued = schedule(w, "c1")  # waits behind the single worker
    finally:
        blocking.release.set()
        w.close(wait_running=False)  # cancels the queued job; the running attempt is left to finish or lease out
    assert value(m, CANCELLED) == 1
    assert s.job_state(queued) == "queued"  # cancellation is not failure: the job survives in the store

    # A new process starts: recovery resubmits the queued job, which had been orphaned before this fix.
    m2 = MemoryMetrics(CollectorRegistry())
    w2 = CompactionWorker(s, ExtractiveSummarizer(), workers=1, metrics=m2)
    try:
        assert queued in w2.recover()
        assert w2.drain(10)
    finally:
        w2.close()
    assert s.job_state(queued) == "done"
    assert value(m2, SERVED) >= 1


def test_closed_worker_refuses_new_work_and_counts_nothing_more():
    s = store_with()
    m = MemoryMetrics(CollectorRegistry())
    w = CompactionWorker(s, ExtractiveSummarizer(), workers=1, metrics=m)
    w.close()
    with pytest.raises(RuntimeError, match="closed"):
        schedule(w, "c0")
    assert value(m, CANCELLED) == 0


def test_stale_running_job_is_requeued_and_its_lease_is_honoured():
    now = [1000.0]
    s = ConversationStore(":memory:", cipher=TenantCipher(KEY), clock=lambda: now[0])
    s.create_conversation("acme", "c0")
    for i in range(4):
        s.append_turn("acme", "c0", "user" if i % 2 == 0 else "assistant", f"decision {i}")
    job = s.plan_compaction("acme", "c0", threshold_tokens=0, keep_recent_turns=1)
    assert s.claim_job(job, lease_seconds=60) is not None  # a worker that then died
    now[0] += 61
    m = MemoryMetrics(CollectorRegistry())
    w = CompactionWorker(s, ExtractiveSummarizer(), workers=1, lease_seconds=60, metrics=m)
    try:
        assert job in w.recover()
        assert w.drain(10)
    finally:
        w.close()
    assert s.job_state(job) == "done"


# ---- registration and exposure -----------------------------------------------------------------------

def test_repeated_initialisation_never_registers_the_same_series_twice():
    assert default_metrics() is default_metrics()  # one process-wide instance
    s = store_with()
    for _ in range(3):  # re-creating workers, as a reloaded app or a test fixture would
        CompactionWorker(s, ExtractiveSummarizer(), workers=1).close()
    fresh = MemoryMetrics(CollectorRegistry())
    with pytest.raises(ValueError, match="Duplicated"):  # the same registry twice is what the default avoids
        MemoryMetrics(fresh.registry)


def test_the_series_have_their_documented_types_and_no_labels_besides_histogram_buckets():
    s = store_with()
    m = MemoryMetrics(CollectorRegistry())
    w = CompactionWorker(s, ExtractiveSummarizer(), workers=1, metrics=m)
    try:
        schedule(w, "c0")
        w.drain(10)
    finally:
        w.close()
    from prometheus_client import generate_latest

    text = generate_latest(m.registry).decode()
    assert "# TYPE contextrag_memory_queue_depth gauge" in text
    assert "# TYPE contextrag_memory_compaction_duration_seconds histogram" in text
    assert "# TYPE contextrag_memory_compaction_jobs_served_total counter" in text
    assert "# TYPE contextrag_memory_compaction_jobs_failed_total counter" in text
    for line in text.splitlines():
        if line.startswith("contextrag_memory") and "{" in line:
            assert line.split("{", 1)[1].startswith('le="'), line  # the only label is the histogram bucket bound


def test_the_metrics_endpoint_exposes_the_worker_series_next_to_the_existing_counters():
    s = store_with()
    rag = offline()
    rag.count('queries_total{status="verified"}', 2)
    rag.memory = s
    m = MemoryMetrics(CollectorRegistry())
    rag.compactor = CompactionWorker(s, ExtractiveSummarizer(), workers=1, metrics=m)
    try:
        schedule(rag.compactor, "c0")
        assert rag.compactor.drain(10)
        client = TestClient(create_app(rag, keystore=None, rate_limit_per_min=None))
        body = client.get("/metrics").text
    finally:
        rag.close()
    assert 'contextrag_queries_total{status="verified"} 2' in body
    for name in (DEPTH, "contextrag_memory_compaction_duration_seconds", SERVED, FAILED):
        assert name in body
    assert f"{SERVED} 1.0" in body


def test_a_pipeline_without_a_compactor_exposes_only_the_existing_counters():
    rag = offline()
    try:
        assert "contextrag_memory" not in rag.metrics_text()
    finally:
        rag.close()
