"""Tenant-scoped PostgreSQL access for the retrieval path.

Connections come from a pool opened as the *application role* (``POSTGRES_DSN``, e.g. ``contextrag_app``): not a
superuser, not ``BYPASSRLS`` and not the owner of the tables. Every unit of work runs in a transaction whose first
statement is ``set_config('app.tenant', <tenant>, true)`` (the transaction-local equivalent of ``SET LOCAL``); the
``tenant_isolation`` row-level-security policies then expose only that tenant's rows.

``retrieval_audit`` is RANGE-partitioned by month. Partitions are created with the admin connection
(``POSTGRES_ADMIN_DSN``, the owner); the application role never needs DDL rights.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
import time
from typing import TYPE_CHECKING

from . import retention

if TYPE_CHECKING:
    from psycopg import Connection

log = logging.getLogger("contextrag.db")

TENANT_SETTING = "app.tenant"  # the name the RLS policies read; the same one contextrag.audit uses
TABLE = "retrieval_audit"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id uuid NOT NULL,
    ts timestamptz NOT NULL DEFAULT now(),
    tenant text NOT NULL,
    user_id text,
    query text NOT NULL,
    top_k integer NOT NULL,
    fragments integer NOT NULL,
    doc_ids jsonb NOT NULL,
    latency_ms double precision NOT NULL,
    PRIMARY KEY (id, ts)
) PARTITION BY RANGE (ts);
CREATE INDEX IF NOT EXISTS {TABLE}_tenant_ts ON {TABLE} (tenant, ts DESC);
ALTER TABLE {TABLE} ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON {TABLE};
CREATE POLICY tenant_isolation ON {TABLE}
    USING (tenant = current_setting('{TENANT_SETTING}', true))
    WITH CHECK (tenant = current_setting('{TENANT_SETTING}', true));
"""


def _scalar(row: tuple | None) -> str:
    if row is None:
        raise RuntimeError("query returned no row")
    return str(row[0])


class TenantDB:
    """Pool of application-role connections; every transaction is bound to one tenant."""

    def __init__(self, app_dsn: str, admin_dsn: str | None = None, *, min_size: int = 1, max_size: int = 10,
                 timeout: float = 5.0):
        import psycopg
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(app_dsn, min_size=min_size, max_size=max_size, timeout=timeout, open=True,
                                   kwargs={"autocommit": True})
        self._admin_lock = threading.Lock()
        self._admin = psycopg.connect(admin_dsn or app_dsn, autocommit=True)
        self._known_months: set[str] = set()
        try:
            with self.pool.connection() as conn:
                self.app_role = _scalar(conn.execute("SELECT current_user").fetchone())
            self._ensure_schema()
        except Exception:  # a refused schema must not leave pool workers running behind the caller
            self.close()
            raise

    def _ensure_schema(self) -> None:
        from psycopg import sql

        with self._admin_lock, self._admin.transaction():
            # several API workers may start at once: serialize the DDL across processes too
            self._admin.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"contextrag.schema.{TABLE}",))
            row = self._admin.execute("SELECT relkind FROM pg_class WHERE oid = to_regclass(%s)", (TABLE,)).fetchone()
            if row and row[0] != "p":
                raise RuntimeError(f"{TABLE} exists but is not partitioned; apply "
                                   "deploy/migrations/001_retrieval_audit_partitioned.sql first")
            self._admin.execute(_SCHEMA)
            owner = _scalar(self._admin.execute("SELECT current_user").fetchone())
            if self.app_role != owner:  # append-only for the application: no UPDATE/DELETE grant
                self._admin.execute(sql.SQL("GRANT SELECT, INSERT ON {} TO {}").format(
                    sql.Identifier(TABLE), sql.Identifier(self.app_role)))
            retention.ensure_partitions(self._admin, TABLE, months_ahead=2)

    def _ensure_partitions_for_now(self) -> None:
        """Make sure the partition for the current month exists before an insert needs it. Cached per month, so
        the DDL runs once a month, not once per request."""
        key = retention.partition_name(TABLE, retention.month_of(time.time()))
        if key in self._known_months:
            return
        with self._admin_lock:
            retention.ensure_partitions(self._admin, TABLE, months_ahead=1)
            self._known_months.add(key)

    @contextmanager
    def tenant_transaction(self, tenant: str) -> Iterator[Connection]:
        """Transaction in which RLS exposes only ``tenant``'s rows. Committed on success, rolled back on error."""
        if not tenant:
            raise ValueError("tenant is required")
        self._ensure_partitions_for_now()
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT set_config(%s, %s, true)", (TENANT_SETTING, tenant))
            yield conn

    def log_retrieval(self, conn: Connection, record: dict) -> None:
        """Append one retrieval record inside the caller's tenant transaction (RLS WITH CHECK enforces the tenant)."""
        from psycopg import sql

        stmt = sql.SQL("""INSERT INTO {t} (id, tenant, user_id, query, top_k, fragments, doc_ids, latency_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)""").format(t=sql.Identifier(TABLE))
        conn.execute(
            stmt,
            (record["id"], record["tenant"], record["user"], record["query"], record["top_k"],
             record["fragments"], json.dumps(record["doc_ids"]), record["latency_ms"]))

    def retention(self, months: int, *, dry_run: bool = False) -> list[retention.TableRetention]:
        """Apply the retention policy to the audit tables managed by this process (admin connection)."""
        with self._admin_lock:
            result = retention.run_retention(self._admin, months, dry_run=dry_run)
            if not dry_run:
                self._known_months.clear()  # dropped partitions must be recreated if needed again
        return result

    def health(self) -> bool:
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.pool.close()
        if not self._admin.closed:
            self._admin.close()

    @classmethod
    def from_env(cls) -> TenantDB | None:
        dsn = os.getenv("POSTGRES_DSN")
        if not dsn:
            return None
        return cls(dsn, os.getenv("POSTGRES_ADMIN_DSN") or None)

