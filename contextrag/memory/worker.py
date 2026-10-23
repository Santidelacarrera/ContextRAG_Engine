"""Bounded, restart-safe background compaction.

* Concurrency is bounded by a fixed thread pool; the number of pending jobs is bounded by the store's queue limit.
* Duplicate scheduling is suppressed twice: the job id is derived from (tenant, conversation, through_seq), and an
  in-process registry refuses to submit a job that is already running.
* Failures are recorded with an error code only (never message content) and retried up to the store's attempt limit.
* Durability: all state lives in the store. After a crash, :meth:`recover` requeues jobs whose lease expired and
  resubmits every queued job, so a restart resumes the work instead of losing it. Shutdown cancels work that has not
  started; that work stays ``queued`` in the store and :meth:`recover` picks it up. Running work is either finished or
  left to lease recovery. Neither path can commit a partial summary, because a commit is one transaction.
* Observability: see :mod:`contextrag.memory.metrics` for the Prometheus series this worker maintains.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

from .metrics import MemoryMetrics, default_metrics
from .store import ConversationStore
from .summarizer import Summarizer

log = logging.getLogger("contextrag.memory")


class CompactionWorker:
    def __init__(self, store: ConversationStore, summarizer: Summarizer, *, workers: int = 2,
                 lease_seconds: float = 300.0, metrics: MemoryMetrics | None = None):
        if workers < 1 or lease_seconds <= 0:
            raise ValueError("workers must be >= 1 and lease_seconds > 0")
        self._store = store
        self._summarizer = summarizer
        self._lease = lease_seconds
        self.metrics = metrics if metrics is not None else default_metrics()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="memory")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._sync_depth()

    def schedule(self, tenant: str, conversation_id: str, *, threshold_tokens: int, keep_recent_turns: int) -> str | None:
        """Plan a compaction if the history is over threshold and submit it. Never blocks on summarization."""
        if self._closed:
            raise RuntimeError("worker is closed")
        job_id = self._store.plan_compaction(tenant, conversation_id, threshold_tokens=threshold_tokens,
                                             keep_recent_turns=keep_recent_turns)
        if job_id is not None:
            self._submit(job_id)
            self._sync_depth()
        return job_id

    def recover(self) -> list[str]:
        """Requeue jobs whose lease expired, then resubmit every queued job. Call once at startup.

        Returns the ids that were submitted to this process."""
        self._store.requeue_stale(lease_seconds=self._lease)
        ids = self._store.queued_jobs()
        for job_id in ids:
            self._submit(job_id)
        self._sync_depth()
        return ids

    def drain(self, timeout: float = 30.0) -> bool:
        """Wait until no job is running and none is pending in the store. Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = [f for f in self._futures.values() if not f.done()]
            if not pending and not self._store.active_jobs():
                return True
            if pending:
                wait(pending, timeout=min(0.05, max(0.0, deadline - time.monotonic())))
            else:
                time.sleep(0.01)  # jobs exist in the store with no future: a retry is about to be submitted
        return False

    def close(self, *, wait_running: bool = True) -> None:
        """Stop accepting work. Queued jobs that have not started are cancelled (they stay queued in the store);
        running attempts finish when ``wait_running`` is true, otherwise they are left to lease recovery."""
        with self._lock:
            self._closed = True
            futures = list(self._futures.values())
        self._pool.shutdown(wait=wait_running, cancel_futures=True)
        cancelled = sum(1 for f in futures if f.cancelled())
        if cancelled:
            self.metrics.cancelled.inc(cancelled)
            log.info("compaction shutdown cancelled=%d (left queued for recovery)", cancelled)
        self._sync_depth()

    def _submit(self, job_id: str) -> None:
        with self._lock:
            if self._closed:
                return
            current = self._futures.get(job_id)
            if current is not None and not current.done():
                return  # already running or queued in this process: no duplicate
            self._futures[job_id] = self._pool.submit(self._run, job_id)

    def _resubmit(self, job_id: str) -> None:
        """Retry from inside the running attempt. The running future is still ``not done`` here, so the duplicate
        check in ``_submit`` would refuse its own retry; this path replaces the registry entry on purpose."""
        with self._lock:
            if not self._closed:
                self._futures[job_id] = self._pool.submit(self._run, job_id)

    def _sync_depth(self) -> None:
        self.metrics.queue_depth.set(self._store.count_active())

    def _run(self, job_id: str) -> None:
        claim = self._store.claim_job(job_id, lease_seconds=self._lease)
        if claim is None:
            self._sync_depth()
            return
        started = time.monotonic()
        try:
            drafts = self._summarizer.summarize(self._store.job_turns(claim))
            committed = self._store.complete_job(claim, drafts)
        except Exception as exc:  # recorded as a code: exception text may contain conversation content
            self.metrics.duration.observe(time.monotonic() - started)
            state = self._store.fail_job(claim, type(exc).__name__)
            if state == "failed":
                self.metrics.failed.inc()
            elif state == "queued":
                self.metrics.retries.inc()
            log.warning("compaction attempt failed job=%s attempt=%d error=%s state=%s",
                        job_id, claim.attempt, type(exc).__name__, state)
            if state == "queued":
                self._resubmit(job_id)
        else:
            self.metrics.duration.observe(time.monotonic() - started)
            if committed:
                self.metrics.served.inc()
        finally:
            self._sync_depth()
