"""Tenant-scoped PostgreSQL access for the retrieval path.

Connections come from a pool opened as the *application role* (``POSTGRES_DSN``, e.g. ``contextrag_app``): not a
superuser, not ``BYPASSRLS`` and not the owner of the tables. Every unit of work runs in a transaction whose first
statement is ``set_config('app.tenant', <tenant>, true)`` (the transaction-local equivalent of ``SET LOCAL``); the
``tenant_isolation`` row-level-security policies then expose only that tenant's rows.

Schema changes (``retrieval_audit`` and its policy) run on the admin connection (``POSTGRES_ADMIN_DSN``, the owner).
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import Connection

log = logging.getLogger("contextrag.db")

TENANT_SETTING = "app.tenant"  # the name the RLS policies read; the same one contextrag.audit uses

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS retrieval_audit (
    id uuid PRIMARY KEY,
    ts timestamptz NOT NULL DEFAULT now(),
    tenant text NOT NULL,
    user_id text,
    query text NOT NULL,
    top_k integer NOT NULL,
    fragments integer NOT NULL,
    doc_ids jsonb NOT NULL,
    latency_ms double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS retrieval_audit_tenant_ts ON retrieval_audit (tenant, ts DESC);
ALTER TABLE retrieval_audit ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation ON retrieval_audit;
CREATE POLICY tenant_isolation ON retrieval_audit
    USING (tenant = current_setting('{TENANT_SETTING}', true))
    WITH CHECK (tenant = current_setting('{TENANT_SETTING}', true));
"""


class TenantDB:
    """Pool of application-role connections; every transaction is bound to one tenant."""

    def __init__(self, app_dsn: str, admin_dsn: str | None = None, *, min_size: int = 1, max_size: int = 10,
                 timeout: float = 5.0):
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(app_dsn, min_size=min_size, max_size=max_size, timeout=timeout, open=True,
                                   kwargs={"autocommit": True})
        with self.pool.connection() as conn:
            self.app_role = conn.execute("SELECT current_user").fetchone()[0]
        self._ensure_schema(admin_dsn or app_dsn)

    def _ensure_schema(self, admin_dsn: str) -> None:
        import psycopg
        from psycopg import sql

        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(_SCHEMA)
            owner = admin.execute("SELECT current_user").fetchone()[0]
            if self.app_role != owner:  # append-only for the application: no UPDATE/DELETE grant
                admin.execute(sql.SQL("GRANT SELECT, INSERT ON retrieval_audit TO {}")
                              .format(sql.Identifier(self.app_role)))

    @contextmanager
    def tenant_transaction(self, tenant: str) -> Iterator[Connection]:
        """Transaction in which RLS exposes only ``tenant``'s rows. Committed on success, rolled back on error."""
        if not tenant:
            raise ValueError("tenant is required")
        with self.pool.connection() as conn, conn.transaction():
            conn.execute("SELECT set_config(%s, %s, true)", (TENANT_SETTING, tenant))
            yield conn

    def log_retrieval(self, conn: Connection, record: dict) -> None:
        """Append one retrieval record inside the caller's tenant transaction (RLS WITH CHECK enforces the tenant)."""
        import json

        conn.execute(
            """INSERT INTO retrieval_audit (id, tenant, user_id, query, top_k, fragments, doc_ids, latency_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)""",
            (record["id"], record["tenant"], record["user"], record["query"], record["top_k"],
             record["fragments"], json.dumps(record["doc_ids"]), record["latency_ms"]))

    def health(self) -> bool:
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.pool.close()

    @classmethod
    def from_env(cls) -> TenantDB | None:
        dsn = os.getenv("POSTGRES_DSN")
        if not dsn:
            return None
        return cls(dsn, os.getenv("POSTGRES_ADMIN_DSN") or None)
