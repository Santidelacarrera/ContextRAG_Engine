"""Audit trail of every query: question, verdict, per-claim evidence coordinates and sources."""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Protocol

from .core.models import Answer

log = logging.getLogger("contextrag.audit")

TENANT_SETTING = "app.tenant"
_PARTITION = re.compile(r"^answer_audit_(\d{4})(\d{2})$")


def record_of(tenant: str, user: str | None, answer: Answer) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "tenant": tenant,
        "user": user,
        "question": answer.question,
        "status": answer.status.value,
        "answer": answer.text,
        "confidence": answer.confidence,
        "attempts": answer.attempts,
        "reason": answer.reason,
        "cached": answer.cached,
        "claims": [asdict(c) for c in answer.claims],
        "sources": [{"index": c.index, "doc_id": c.parent.doc_id, "source": c.parent.source,
                     "start": c.parent.start, "end": c.parent.end, "page": c.parent.page,
                     "section": c.parent.section, "score": c.score} for c in answer.sources],
    }


class AuditLog(Protocol):
    def write(self, record: dict, query_embedding: list[float] | None = None) -> None: ...

    def list(self, tenant: str, limit: int = 50) -> list[dict]: ...

    def delete_tenant(self, tenant: str) -> int:
        """Erase every audit record of a tenant; returns how many were removed."""
        ...

    def purge(self, older_than_seconds: float) -> int:
        """Retention: erase records older than the given age (all tenants); returns how many were removed."""
        ...

    def health(self) -> bool: ...


class InMemoryAudit:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, record, query_embedding=None):
        self.records.append(record)

    def list(self, tenant, limit=50):
        return [r for r in reversed(self.records) if r["tenant"] == tenant][:limit]

    def delete_tenant(self, tenant):
        keep = [r for r in self.records if r["tenant"] != tenant]
        n, self.records = len(self.records) - len(keep), keep
        return n

    def purge(self, older_than_seconds):
        cutoff = time.time() - older_than_seconds
        keep = [r for r in self.records if r["ts"] >= cutoff]
        n, self.records = len(self.records) - len(keep), keep
        return n

    def health(self):
        return True


class PostgresAudit:
    """Audit log in PostgreSQL 16: monthly range partitions and row-level security per tenant.

    Isolation is enforced by the database, not only by the application. Every statement on the application
    connection (``dsn``) runs in a transaction that sets ``app.tenant`` (transaction-local), and the policy
    ``tenant_isolation`` exposes only the rows of that tenant. Schema, partitions and retention run on the
    admin connection (``admin_dsn``, the table owner, which RLS does not restrict).

    Superusers, ``BYPASSRLS`` roles and the table owner ignore RLS, so ``dsn`` must be a plain role;
    ``rls_enforced`` reports whether it is. If the ``vector`` extension is available the query embedding is
    stored as ``vector(dim)`` (pgvector, enabling similar-query lookups); otherwise as ``float8[]``.
    """

    def __init__(self, dsn: str, dim: int, admin_dsn: str | None = None):
        import psycopg

        self.dim = int(dim)
        self._lock = threading.RLock()  # one transaction at a time per connection
        self._partitions: set[str] = set()
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.admin = psycopg.connect(admin_dsn, autocommit=True) if admin_dsn else self.conn
        self.has_vector = self._create_extension()
        self._create_schema()
        self.rls_enforced = self._rls_enforced()
        if not self.rls_enforced:
            log.warning("audit: the application role bypasses row-level security (superuser, BYPASSRLS or table "
                        "owner); tenant isolation then relies on the application alone. Use a dedicated role.")

    # -- schema -----------------------------------------------------------------------------
    def _create_extension(self) -> bool:
        try:
            self.admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
            return True
        except Exception:
            return False

    def _create_schema(self) -> None:
        from psycopg import sql

        col = f"vector({self.dim})" if self.has_vector else "double precision[]"
        with self._lock:
            row = self.admin.execute("SELECT relkind FROM pg_class WHERE oid = to_regclass('answer_audit')").fetchone()
            if row and row[0] != "p":
                raise RuntimeError("answer_audit exists but is not partitioned; migrate it before starting")
            self.admin.execute(f"""
                CREATE TABLE IF NOT EXISTS answer_audit (
                    id uuid NOT NULL,
                    ts timestamptz NOT NULL DEFAULT now(),
                    tenant text NOT NULL,
                    user_id text,
                    question text NOT NULL,
                    status text NOT NULL CHECK (status IN ('verified', 'fallback')),
                    answer text NOT NULL,
                    confidence double precision NOT NULL,
                    attempts integer NOT NULL,
                    reason text NOT NULL DEFAULT '',
                    cached boolean NOT NULL DEFAULT false,
                    claims jsonb NOT NULL,
                    sources jsonb NOT NULL,
                    query_embedding {col},
                    PRIMARY KEY (id, ts)
                ) PARTITION BY RANGE (ts)""")
            self.admin.execute("CREATE INDEX IF NOT EXISTS answer_audit_tenant_ts ON answer_audit (tenant, ts DESC)")
            self.admin.execute("ALTER TABLE answer_audit ENABLE ROW LEVEL SECURITY")
            self.admin.execute("DROP POLICY IF EXISTS tenant_isolation ON answer_audit")
            self.admin.execute(
                f"CREATE POLICY tenant_isolation ON answer_audit "
                f"USING (tenant = current_setting('{TENANT_SETTING}', true)) "
                f"WITH CHECK (tenant = current_setting('{TENANT_SETTING}', true))")
            app_role = self.conn.execute("SELECT current_user").fetchone()[0]
            if app_role != self.admin.execute("SELECT current_user").fetchone()[0]:
                self.admin.execute(sql.SQL("GRANT SELECT, INSERT, DELETE ON answer_audit TO {}")
                                   .format(sql.Identifier(app_role)))
            self.ensure_partitions()

    def _rls_enforced(self) -> bool:
        row = self.conn.execute("""
            SELECT NOT (r.rolsuper OR r.rolbypassrls OR c.relowner = r.oid)
            FROM pg_roles r, pg_class c
            WHERE r.rolname = current_user AND c.oid = to_regclass('answer_audit')""").fetchone()
        return bool(row and row[0])

    # -- partitions -------------------------------------------------------------------------
    @staticmethod
    def _month_of(ts: float) -> tuple[str, datetime, datetime]:
        start = datetime.fromtimestamp(ts, tz=timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return f"answer_audit_{start:%Y%m}", start, (start + timedelta(days=32)).replace(day=1)

    def ensure_partition(self, ts: float) -> None:
        """Create the monthly partition that holds ``ts`` if it does not exist yet."""
        from psycopg import sql

        name, start, end = self._month_of(ts)
        with self._lock:
            if name in self._partitions:
                return
            self.admin.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {} PARTITION OF answer_audit "
                                       "FOR VALUES FROM ({}) TO ({})")
                               .format(sql.Identifier(name), sql.Literal(start), sql.Literal(end)))
            self._partitions.add(name)

    def ensure_partitions(self, months_ahead: int = 2) -> None:
        """Create the partitions for the current month and the next ``months_ahead`` months."""
        _, start, _ = self._month_of(time.time())
        for _ in range(months_ahead + 1):
            self.ensure_partition(start.timestamp())
            start = (start + timedelta(days=32)).replace(day=1)

    # -- tenant-scoped transactions ---------------------------------------------------------
    @contextmanager
    def _as_tenant(self, tenant: str) -> Iterator:
        """A transaction in which RLS exposes only ``tenant``'s rows; committed on success, rolled back on error."""
        with self._lock, self.conn.transaction():
            self.conn.execute("SELECT set_config(%s, %s, true)", (TENANT_SETTING, tenant))
            yield self.conn

    # -- AuditLog ---------------------------------------------------------------------------
    def write(self, record, query_embedding=None):
        emb = None
        if query_embedding is not None:
            emb = "[" + ",".join(map(str, query_embedding)) + "]" if self.has_vector else list(query_embedding)
        cast = "%s::vector" if self.has_vector else "%s"
        self.ensure_partition(record["ts"])
        with self._as_tenant(record["tenant"]) as conn:
            conn.execute(
                f"""INSERT INTO answer_audit (id, ts, tenant, user_id, question, status, answer, confidence,
                        attempts, reason, cached, claims, sources, query_embedding)
                    VALUES (%s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, {cast})""",
                (record["id"], record["ts"], record["tenant"], record["user"], record["question"], record["status"],
                 record["answer"], record["confidence"], record["attempts"], record["reason"], record["cached"],
                 json.dumps(record["claims"]), json.dumps(record["sources"]), emb))

    def list(self, tenant, limit=50):
        with self._as_tenant(tenant) as conn:
            rows = conn.execute(
                """SELECT id, extract(epoch from ts), user_id, question, status, answer, confidence, attempts, reason,
                          cached, claims, sources
                   FROM answer_audit WHERE tenant = %s ORDER BY ts DESC LIMIT %s""", (tenant, limit)).fetchall()
        keys = ("id", "ts", "user", "question", "status", "answer", "confidence", "attempts", "reason",
                "cached", "claims", "sources")
        return [{**dict(zip(keys, r, strict=True)), "id": str(r[0]), "tenant": tenant} for r in rows]

    def delete_tenant(self, tenant):
        with self._as_tenant(tenant) as conn:
            return conn.execute("DELETE FROM answer_audit WHERE tenant = %s", (tenant,)).rowcount

    def purge(self, older_than_seconds):
        """Retention across tenants (admin connection). Whole months that expired are dropped in one step; the
        boundary month is deleted row by row. Returns the number of records removed."""
        from psycopg import sql

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=float(older_than_seconds))
        removed = 0
        with self._lock:
            names = [r[0] for r in self.admin.execute(
                "SELECT c.relname FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
                "WHERE i.inhparent = to_regclass('answer_audit')").fetchall()]
            for name in names:
                m = _PARTITION.match(name)
                if not m:
                    continue
                year, month = int(m.group(1)), int(m.group(2))
                end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=timezone.utc)
                if end > cutoff:
                    continue
                removed += self.admin.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(name))).fetchone()[0]
                self.admin.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(name)))
                self._partitions.discard(name)
            removed += self.admin.execute("DELETE FROM answer_audit WHERE ts < %s", (cutoff,)).rowcount
        return removed

    def health(self):
        try:
            self.conn.execute("SELECT 1")
            return True
        except Exception:
            return False
