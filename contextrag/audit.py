"""Audit trail of every query: question, verdict, per-claim evidence coordinates and sources."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Protocol

from . import retention
from .core.models import Answer

log = logging.getLogger("contextrag.audit")

TENANT_SETTING = "app.tenant"


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
        with self._lock, self.admin.transaction():
            self.admin.execute("SELECT pg_advisory_xact_lock(hashtext('contextrag.schema.answer_audit'))")
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
    def ensure_partition(self, ts: float) -> None:
        """Create the monthly partition that holds ``ts`` if it does not exist yet (one DDL per month)."""
        name = retention.partition_name("answer_audit", retention.month_of(ts))
        with self._lock:
            if name not in self._partitions:
                retention.ensure_partition(self.admin, "answer_audit", ts)
                self._partitions.add(name)

    def ensure_partitions(self, months_ahead: int = 2) -> None:
        """Create the partitions for the current month and the next ``months_ahead`` months."""
        with self._lock:
            retention.ensure_partitions(self.admin, "answer_audit", months_ahead)

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
        """Retention across tenants (admin connection): whole expired months are dropped, the rest deleted row by
        row. Returns the number of records removed."""
        with self._lock:
            result = retention.apply_retention(self.admin, "answer_audit",
                                               retention.older_than(float(older_than_seconds)))
            self._partitions.difference_update(result.dropped_partitions)
        return result.rows_removed

    def health(self):
        try:
            self.conn.execute("SELECT 1")
            return True
        except Exception:
            return False
