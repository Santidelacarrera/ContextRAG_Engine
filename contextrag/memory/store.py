"""Durable conversation store (SQLite, standard library): turns, structured memories and compaction jobs.

Consistency model
-----------------
* Turns are immutable and numbered ``seq = 1, 2, ...`` per conversation. A conversation's ``version`` equals its
  last ``seq``; every append increments it. A writer may pass ``expected_version`` (optimistic concurrency): if the
  conversation moved on, the append is refused with :class:`VersionConflict` instead of interleaving silently.
* A compaction job is identified by ``(tenant, conversation, through_seq)``, and ``through_seq`` is fixed when the job
  is planned. The job summarizes the turns in ``(compacted_through, through_seq]``. Turns are immutable, so the job
  never sees a moving target.
* Committing a job is a single transaction that (a) checks the job still holds its claim, (b) inserts its memories and
  (c) advances ``compacted_through`` **only forward**. A stale or duplicate job therefore cannot overwrite newer state
  and cannot insert memories twice. Turns newer than ``through_seq`` are untouched.
* Source turns are never deleted by compaction. They remain the provenance of every memory (``source_from`` to
  ``source_to``). Deletion happens only through :meth:`delete_conversation`, for erasure.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..core.chunking import count_tokens

ROLES = ("user", "assistant")
MEMORY_ORIGINS = ("user_stated", "model_inferred")
JOB_STATES = ("queued", "running", "done", "failed")
MAX_TURN_CHARS = 20_000


class ConversationNotFound(LookupError):
    pass


class VersionConflict(RuntimeError):
    """The conversation changed since the caller read its version."""

    def __init__(self, expected: int, actual: int):
        super().__init__(f"expected version {expected}, conversation is at {actual}")
        self.expected, self.actual = expected, actual


class QueueFull(RuntimeError):
    """Too many compaction jobs are pending; scheduling is refused rather than queued without bound."""


@dataclass(frozen=True)
class Turn:
    seq: int
    role: str
    content: str


@dataclass(frozen=True)
class Memory:
    memory_id: str
    memory_type: str
    content: str
    origin: str
    confidence: float
    source_from: int
    source_to: int


@dataclass(frozen=True)
class MemoryDraft:
    memory_type: str
    content: str
    origin: str
    confidence: float
    source_seq: int  # the turn the memory was derived from; must lie in the job's range


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    tenant: str
    conversation_id: str
    through_seq: int
    attempt: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    tenant TEXT NOT NULL, conversation_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 0,
    compacted_through INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (tenant, conversation_id));
CREATE TABLE IF NOT EXISTS turns (
    tenant TEXT NOT NULL, conversation_id TEXT NOT NULL, seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL, created_at REAL NOT NULL,
    PRIMARY KEY (tenant, conversation_id, seq));
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, conversation_id TEXT NOT NULL,
    memory_type TEXT NOT NULL, content TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('user_stated', 'model_inferred')),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    source_from INTEGER NOT NULL, source_to INTEGER NOT NULL,
    job_id TEXT NOT NULL, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS memories_conv ON memories (tenant, conversation_id);
CREATE TABLE IF NOT EXISTS compaction_jobs (
    job_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, conversation_id TEXT NOT NULL,
    through_seq INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('queued', 'running', 'done', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0, claimed_at REAL, error TEXT, created_at REAL NOT NULL,
    UNIQUE (tenant, conversation_id, through_seq));
CREATE INDEX IF NOT EXISTS jobs_state ON compaction_jobs (state);
"""


class ConversationStore:
    def __init__(self, path: str | Path = ":memory:", *, max_attempts: int = 3, queue_limit: int = 64,
                 clock: Callable[[], float] = time.time):
        if max_attempts < 1 or queue_limit < 1:
            raise ValueError("max_attempts and queue_limit must be >= 1")
        self._path = str(path)
        self._max_attempts, self._queue_limit, self._clock = max_attempts, queue_limit, clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA busy_timeout = 30000")
        if self._path != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")  # durable across restarts, readers do not block writers
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    # -- conversations and turns ----------------------------------------------------------------------
    def create_conversation(self, tenant: str, conversation_id: str) -> None:
        _require_id(tenant, "tenant")
        _require_id(conversation_id, "conversation_id")
        with self._tx() as db:
            db.execute("INSERT OR IGNORE INTO conversations (tenant, conversation_id, created_at) VALUES (?, ?, ?)",
                       (tenant, conversation_id, self._clock()))

    def version(self, tenant: str, conversation_id: str) -> int:
        return self._conversation(tenant, conversation_id)["version"]

    def append_turn(self, tenant: str, conversation_id: str, role: str, content: str, *,
                    expected_version: int | None = None) -> int:
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        content = content.strip()
        if not content or len(content) > MAX_TURN_CHARS:
            raise ValueError(f"turn content must be 1..{MAX_TURN_CHARS} characters")
        with self._tx() as db:
            row = db.execute("SELECT version FROM conversations WHERE tenant = ? AND conversation_id = ?",
                             (tenant, conversation_id)).fetchone()
            if row is None:
                raise ConversationNotFound(conversation_id)
            current = row[0]
            if expected_version is not None and expected_version != current:
                raise VersionConflict(expected_version, current)
            seq = current + 1
            db.execute("INSERT INTO turns (tenant, conversation_id, seq, role, content, created_at) "
                       "VALUES (?, ?, ?, ?, ?, ?)", (tenant, conversation_id, seq, role, content, self._clock()))
            db.execute("UPDATE conversations SET version = ? WHERE tenant = ? AND conversation_id = ?",
                       (seq, tenant, conversation_id))
        return seq

    def turns_between(self, tenant: str, conversation_id: str, after_seq: int, through_seq: int) -> list[Turn]:
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, role, content FROM turns WHERE tenant = ? AND conversation_id = ? "
                "AND seq > ? AND seq <= ? ORDER BY seq", (tenant, conversation_id, after_seq, through_seq)).fetchall()
        return [Turn(*r) for r in rows]

    def history_window(self, tenant: str, conversation_id: str, token_budget: int,
                       token_fn: Callable[[str], int] = count_tokens) -> list[Turn]:
        """The newest turns that fit ``token_budget``, returned in chronological order."""
        if token_budget < 0:
            raise ValueError("token_budget must be >= 0")
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, role, content FROM turns WHERE tenant = ? AND conversation_id = ? ORDER BY seq DESC",
                (tenant, conversation_id)).fetchall()
        chosen: list[Turn] = []
        used = 0
        for row in rows:
            cost = token_fn(row[2])
            if used + cost > token_budget:
                break  # newest-first: stop at the first turn that does not fit, keeping the window contiguous
            chosen.append(Turn(*row))
            used += cost
        return list(reversed(chosen))

    def uncompacted_tokens(self, tenant: str, conversation_id: str,
                           token_fn: Callable[[str], int] = count_tokens) -> int:
        conv = self._conversation(tenant, conversation_id)
        return sum(token_fn(t.content) for t in self.turns_between(tenant, conversation_id,
                                                                   conv["compacted_through"], conv["version"]))

    # -- compaction jobs ------------------------------------------------------------------------------
    def plan_compaction(self, tenant: str, conversation_id: str, *, threshold_tokens: int, keep_recent_turns: int,
                        token_fn: Callable[[str], int] = count_tokens) -> str | None:
        """Schedule a job covering everything except the most recent turns, if the uncompacted history is over
        the threshold. Idempotent: the same range always yields the same job id."""
        conv = self._conversation(tenant, conversation_id)
        through = conv["version"] - keep_recent_turns
        if through <= conv["compacted_through"]:
            return None
        pending = self.turns_between(tenant, conversation_id, conv["compacted_through"], conv["version"])
        if sum(token_fn(t.content) for t in pending) <= threshold_tokens:
            return None
        job_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant}|{conversation_id}|{through}"))
        with self._tx() as db:
            active = db.execute("SELECT COUNT(*) FROM compaction_jobs WHERE state IN ('queued', 'running')").fetchone()[0]
            exists = db.execute("SELECT 1 FROM compaction_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not exists and active >= self._queue_limit:
                raise QueueFull(f"{active} compaction jobs pending (limit {self._queue_limit})")
            db.execute("INSERT OR IGNORE INTO compaction_jobs (job_id, tenant, conversation_id, through_seq, state, "
                       "created_at) VALUES (?, ?, ?, ?, 'queued', ?)",
                       (job_id, tenant, conversation_id, through, self._clock()))
        return job_id

    def claim_job(self, job_id: str, *, lease_seconds: float) -> ClaimedJob | None:
        """Take ownership of a queued job, or of a running job whose lease has expired. Returns ``None`` if the job
        is done, failed, or held by a live worker."""
        now = self._clock()
        with self._tx() as db:
            row = db.execute("SELECT tenant, conversation_id, through_seq, state, attempts, claimed_at "
                             "FROM compaction_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                return None
            tenant, cid, through, state, attempts, claimed_at = row
            expired = state == "running" and claimed_at is not None and now - claimed_at >= lease_seconds
            if not (state == "queued" or expired) or attempts >= self._max_attempts:
                return None
            db.execute("UPDATE compaction_jobs SET state = 'running', attempts = attempts + 1, claimed_at = ? "
                       "WHERE job_id = ?", (now, job_id))
        return ClaimedJob(job_id, tenant, cid, through, attempts + 1)

    def job_turns(self, claim: ClaimedJob) -> list[Turn]:
        """The turns a claimed job must summarize: ``(compacted_through, through_seq]`` as committed right now."""
        conv = self._conversation(claim.tenant, claim.conversation_id)
        return self.turns_between(claim.tenant, claim.conversation_id, conv["compacted_through"], claim.through_seq)

    def complete_job(self, claim: ClaimedJob, drafts: Sequence[MemoryDraft]) -> bool:
        """Commit a job's memories and advance ``compacted_through``. Returns False (and writes nothing) if the claim
        was lost, or if the range is already covered by a newer commit. Both cases are safe to ignore."""
        for d in drafts:
            if not d.memory_type or d.origin not in MEMORY_ORIGINS or not (0.0 <= d.confidence <= 1.0):
                raise ValueError("invalid memory draft")
        with self._tx() as db:
            job = db.execute("SELECT state, attempts FROM compaction_jobs WHERE job_id = ?",
                             (claim.job_id,)).fetchone()
            if job is None or job[0] != "running" or job[1] != claim.attempt:
                return False  # the claim was lost to a newer attempt: that attempt owns the commit
            conv = db.execute("SELECT version, compacted_through FROM conversations WHERE tenant = ? "
                              "AND conversation_id = ?", (claim.tenant, claim.conversation_id)).fetchone()
            if conv is None:
                raise ConversationNotFound(claim.conversation_id)
            _, compacted = conv
            if compacted >= claim.through_seq:  # already covered by a newer or equal commit: nothing to add
                db.execute("UPDATE compaction_jobs SET state = 'done', error = NULL WHERE job_id = ?", (claim.job_id,))
                return False
            lo, hi = compacted, claim.through_seq
            for d in drafts:
                if not (lo < d.source_seq <= hi):
                    raise ValueError(f"memory source {d.source_seq} is outside the job range ({lo}, {hi}]")
                db.execute("INSERT INTO memories (memory_id, tenant, conversation_id, memory_type, content, origin, "
                           "confidence, source_from, source_to, job_id, created_at) "
                           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (str(uuid.uuid4()), claim.tenant, claim.conversation_id, d.memory_type, d.content,
                            d.origin, d.confidence, d.source_seq, d.source_seq, claim.job_id, self._clock()))
            db.execute("UPDATE conversations SET compacted_through = ? WHERE tenant = ? AND conversation_id = ?",
                       (hi, claim.tenant, claim.conversation_id))
            db.execute("UPDATE compaction_jobs SET state = 'done', error = NULL WHERE job_id = ?", (claim.job_id,))
        return True

    def fail_job(self, claim: ClaimedJob, error_code: str) -> str:
        """Record a failed attempt (an error code only, never content). Returns the job's new state: ``queued`` if
        another attempt is allowed, ``failed`` otherwise."""
        with self._tx() as db:
            row = db.execute("SELECT state, attempts FROM compaction_jobs WHERE job_id = ?", (claim.job_id,)).fetchone()
            if row is None or row[0] != "running" or row[1] != claim.attempt:
                return row[0] if row else "failed"
            state = "failed" if claim.attempt >= self._max_attempts else "queued"
            db.execute("UPDATE compaction_jobs SET state = ?, error = ?, claimed_at = NULL WHERE job_id = ?",
                       (state, error_code[:64], claim.job_id))
        return state

    def requeue_stale(self, *, lease_seconds: float) -> list[str]:
        """Recovery after a crash: running jobs whose lease expired go back to ``queued``. Returns their ids."""
        cutoff = self._clock() - lease_seconds
        with self._tx() as db:
            ids = [r[0] for r in db.execute(
                "SELECT job_id FROM compaction_jobs WHERE state = 'running' AND claimed_at < ?", (cutoff,)).fetchall()]
            db.executemany("UPDATE compaction_jobs SET state = 'queued', claimed_at = NULL WHERE job_id = ?",
                           [(i,) for i in ids])
        return ids

    def job_state(self, job_id: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT state FROM compaction_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row[0] if row else None

    def active_jobs(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute(
                "SELECT job_id FROM compaction_jobs WHERE state IN ('queued', 'running')").fetchall()]

    # -- memories -------------------------------------------------------------------------------------
    def memories(self, tenant: str, conversation_id: str) -> list[Memory]:
        with self._lock:
            rows = self._db.execute(
                "SELECT memory_id, memory_type, content, origin, confidence, source_from, source_to FROM memories "
                "WHERE tenant = ? AND conversation_id = ? ORDER BY source_from, memory_id",
                (tenant, conversation_id)).fetchall()
        return [Memory(*r) for r in rows]

    def relevant_memories(self, tenant: str, conversation_id: str, query: str, k: int,
                          score: Callable[[str, str], float]) -> list[Memory]:
        """Only the ``k`` memories that match the query (by ``score``), never the whole store."""
        if k < 0:
            raise ValueError("k must be >= 0")
        ranked = [(score(query, m.content), m) for m in self.memories(tenant, conversation_id)]
        ranked = [(s, m) for s, m in ranked if s > 0]
        ranked.sort(key=lambda sm: (-sm[0], sm[1].source_from, sm[1].memory_id))
        return [m for _, m in ranked[:k]]

    def delete_conversation(self, tenant: str, conversation_id: str) -> None:
        """Erasure: removes the conversation, its turns, memories and jobs for this tenant."""
        with self._tx() as db:
            for table in ("memories", "compaction_jobs", "turns"):
                db.execute(f"DELETE FROM {table} WHERE tenant = ? AND conversation_id = ?", (tenant, conversation_id))
            db.execute("DELETE FROM conversations WHERE tenant = ? AND conversation_id = ?", (tenant, conversation_id))

    def _conversation(self, tenant: str, conversation_id: str) -> dict:
        with self._lock:
            row = self._db.execute("SELECT version, compacted_through FROM conversations WHERE tenant = ? "
                                   "AND conversation_id = ?", (tenant, conversation_id)).fetchone()
        if row is None:
            raise ConversationNotFound(conversation_id)
        return {"version": row[0], "compacted_through": row[1]}


def _require_id(value: str, name: str) -> None:
    if not value or len(value) > 128 or not all(c.isalnum() or c in "-_.:" for c in value):
        raise ValueError(f"{name} must be 1..128 characters of letters, digits, '-', '_', '.' or ':'")


__all__ = ["ConversationStore", "ConversationNotFound", "VersionConflict", "QueueFull", "Turn", "Memory",
           "MemoryDraft", "ClaimedJob"]
