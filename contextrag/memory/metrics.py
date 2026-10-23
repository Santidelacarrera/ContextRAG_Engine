"""Prometheus metrics for the compaction worker.

Names and semantics (all label-free, so cardinality is constant):

* ``contextrag_memory_queue_depth`` (gauge): compaction jobs queued or running in the store, sampled after every
  state change this process makes. It is the store's count, not the in-process futures, so it survives restarts.
* ``contextrag_memory_compaction_duration_seconds`` (histogram): wall time of each attempt that claimed a job, from
  claim to commit or failure, measured with ``time.monotonic``. Retries are observed separately.
* ``contextrag_memory_compaction_jobs_served_total`` (counter): jobs whose attempt committed its memories
  (``complete_job`` returned True). A job whose range was already covered by a newer commit is not counted.
* ``contextrag_memory_compaction_jobs_failed_total`` (counter): **terminal** failures, jobs that exhausted their
  attempt limit. A failed attempt that will be retried is not counted here (see the retries counter).
* ``contextrag_memory_compaction_retries_total`` (counter): failed attempts that were requeued for another try.
* ``contextrag_memory_compaction_cancelled_total`` (counter): jobs that were queued in this process and cancelled at
  shutdown. They stay ``queued`` in the store and are resubmitted by ``CompactionWorker.recover``.

Metrics live in a dedicated registry, so they never collide with other libraries' collectors. The registry is created
once per process (the default factory is a lazy singleton), and an explicit registry can be passed for tests.
"""
from __future__ import annotations

import threading

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

DURATION_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


class MemoryMetrics:
    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry if registry is not None else CollectorRegistry(auto_describe=True)
        self.queue_depth = Gauge("contextrag_memory_queue_depth",
                                 "Compaction jobs queued or running in the conversation store", registry=self.registry)
        self.duration = Histogram("contextrag_memory_compaction_duration_seconds",
                                  "Wall time of one compaction attempt, from claim to commit or failure",
                                  buckets=DURATION_BUCKETS, registry=self.registry)
        self.served = Counter("contextrag_memory_compaction_jobs_served", "Compaction jobs whose memories were committed",
                              registry=self.registry)
        self.failed = Counter("contextrag_memory_compaction_jobs_failed",
                              "Compaction jobs that exhausted their attempts", registry=self.registry)
        self.retries = Counter("contextrag_memory_compaction_retries",
                               "Failed compaction attempts that were requeued", registry=self.registry)
        self.cancelled = Counter("contextrag_memory_compaction_cancelled",
                                 "Queued compaction jobs cancelled at shutdown", registry=self.registry)


# prometheus_client appends "_total" to counter names itself, so the names above render as the exact
# ``*_total`` names the exposition documents (``contextrag_memory_compaction_jobs_served_total`` and so on).

_default: MemoryMetrics | None = None
_default_lock = threading.Lock()


def default_metrics() -> MemoryMetrics:
    """The process-wide instance. Creating it twice would raise on duplicate registration, so it is built once."""
    global _default
    with _default_lock:
        if _default is None:
            _default = MemoryMetrics()
        return _default


__all__ = ["MemoryMetrics", "default_metrics", "DURATION_BUCKETS"]
