"""Operations: retention and partitions, the partitioning migration, the CLI, and pool behaviour under concurrency.

Runs against a real PostgreSQL 16 server (see ``pg_dsn`` in conftest.py); skipped when it is not installed."""
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
psycopg = pytest.importorskip("psycopg")
from fastapi.testclient import TestClient  # noqa: E402

from contextrag import factory, retention  # noqa: E402
from contextrag.api.server import create_app  # noqa: E402
from contextrag.db import TenantDB  # noqa: E402
from conftest import POLICY  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "deploy" / "migrations" / "001_retrieval_audit_partitioned.sql"
QUESTION = "¿Cuánto tiempo se conservan los registros de auditoría?"


def months_ago(k: int, day: int = 15):
    """A mid-month instant k calendar months before now (UTC)."""
    from datetime import datetime, timezone

    return retention.add_months(retention.first_of_month(datetime.now(timezone.utc)), -k).replace(day=day)


def insert_audit(admin, ts, tenant="acme", query="q"):
    retention.ensure_partition(admin, "retrieval_audit", ts.timestamp())
    admin.execute("INSERT INTO retrieval_audit (id, ts, tenant, query, top_k, fragments, doc_ids, latency_ms) "
                  "VALUES (%s, %s, %s, %s, 1, 0, '[]', 1)", (str(uuid.uuid4()), ts, tenant, query))


def count(admin, table="retrieval_audit", where="TRUE", params=()):
    return admin.execute(f"SELECT count(*) FROM {table} WHERE {where}", params).fetchone()[0]


@pytest.fixture(scope="module")
def admin(pg_dsn):
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        yield conn


@contextlib.contextmanager
def tenant_db(pg_dsn, pg_app_dsn, **kw):
    db = TenantDB(pg_app_dsn, admin_dsn=pg_dsn, **kw)
    try:
        yield db
    finally:
        db.close()


# ---- retention ---------------------------------------------------------------------------

def test_retention_drops_expired_months_and_keeps_the_last_n(pg_dsn, pg_app_dsn, admin):
    with tenant_db(pg_dsn, pg_app_dsn):
        admin.execute("DELETE FROM retrieval_audit")
        for k in (5, 4, 3, 2, 1, 0):  # one record per month, the oldest five months ago
            insert_audit(admin, months_ago(k))

        dry = retention.run_retention(admin, 3, dry_run=True, tables=("retrieval_audit",))[0]
        assert dry.dry_run and dry.rows_removed == 3 and len(dry.dropped_partitions) == 3
        assert count(admin) == 6  # a dry run changes nothing

        [real] = retention.run_retention(admin, 3, tables=("retrieval_audit",))
        assert real.rows_removed == 3 and real.dropped_partitions == dry.dropped_partitions
        remaining = [name for name, _ in retention.list_partitions(admin, "retrieval_audit")]
        assert not set(real.dropped_partitions) & set(remaining)
        assert count(admin) == 3  # the current month and the two before it survive


def test_retention_deletes_rows_older_than_the_cutoff_inside_a_kept_month(pg_dsn, pg_app_dsn, admin):
    with tenant_db(pg_dsn, pg_app_dsn):
        admin.execute("DELETE FROM retrieval_audit")
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        insert_audit(admin, now - timedelta(days=40))
        insert_audit(admin, now - timedelta(days=5))
        res = retention.apply_retention(admin, "retrieval_audit", retention.older_than(30 * 86400))
        assert res.rows_removed == 1
        assert count(admin, where="ts > now() - interval '30 days'") == 1


def test_retention_refuses_naive_cutoffs_and_nonsense_windows(admin):
    from datetime import datetime

    with pytest.raises(ValueError, match="timezone-aware"):
        retention.apply_retention(admin, "retrieval_audit", datetime(2020, 1, 1))
    with pytest.raises(ValueError, match="months must be"):
        retention.retention_cutoff(0)


def test_inserts_create_the_current_partition_on_demand_and_survive_retention(pg_dsn, pg_app_dsn, admin):
    with tenant_db(pg_dsn, pg_app_dsn) as db:
        admin.execute(f"DROP TABLE IF EXISTS {retention.partition_name('retrieval_audit', retention.month_of(time.time()))}")
        with db.tenant_transaction("acme") as conn:  # must recreate the current month before inserting
            db.log_retrieval(conn, {"id": str(uuid.uuid4()), "tenant": "acme", "user": None, "query": "hola",
                                    "top_k": 1, "fragments": 0, "doc_ids": [], "latency_ms": 0.1})
        db.retention(months=1)  # drops older months only; the cache must not forget the current one
        with db.tenant_transaction("acme") as conn:
            db.log_retrieval(conn, {"id": str(uuid.uuid4()), "tenant": "acme", "user": None, "query": "adios",
                                    "top_k": 1, "fragments": 0, "doc_ids": [], "latency_ms": 0.1})


# ---- migration 001 ------------------------------------------------------------------------

LEGACY_SCHEMA = """
CREATE TABLE retrieval_audit (
    id uuid PRIMARY KEY, ts timestamptz NOT NULL DEFAULT now(), tenant text NOT NULL, user_id text,
    query text NOT NULL, top_k integer NOT NULL, fragments integer NOT NULL, doc_ids jsonb NOT NULL,
    latency_ms double precision NOT NULL);
CREATE INDEX retrieval_audit_tenant_ts ON retrieval_audit (tenant, ts DESC);
ALTER TABLE retrieval_audit ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON retrieval_audit USING (tenant = current_setting('app.tenant', true))
    WITH CHECK (tenant = current_setting('app.tenant', true));
GRANT SELECT, INSERT ON retrieval_audit TO contextrag_app;
"""


@pytest.mark.skipif(shutil.which("psql") is None, reason="psql not installed")
def test_migration_001_converts_a_plain_table_keeps_rows_and_refuses_to_run_twice(pg_dsn, pg_app_dsn, admin):
    admin.execute("DROP DATABASE IF EXISTS migr_test WITH (FORCE)")
    admin.execute("CREATE DATABASE migr_test")
    base = pg_dsn.rsplit("/", 1)[0]
    migr_admin, migr_app = f"{base}/migr_test", pg_app_dsn.rsplit("/", 1)[0] + "/migr_test"
    with psycopg.connect(migr_admin, autocommit=True) as c:
        c.execute(LEGACY_SCHEMA)
        for k in (0, 3):
            c.execute("INSERT INTO retrieval_audit (id, ts, tenant, query, top_k, fragments, doc_ids, latency_ms) "
                      "VALUES (gen_random_uuid(), %s, 'acme', 'legacy', 1, 0, '[]', 1)", (months_ago(k),))

    # the application refuses to start on the unmigrated schema, and does not leave its pool behind
    with pytest.raises(RuntimeError, match="not partitioned"):
        TenantDB(migr_app, admin_dsn=migr_admin)

    def psql():
        return subprocess.run(["psql", "-v", "ON_ERROR_STOP=1", "-v", "app_role=contextrag_app", "-q", "-f",
                               str(MIGRATION), migr_admin], capture_output=True, text=True)

    assert psql().returncode == 0
    assert psql().returncode != 0  # the guard: already migrated, nothing to do
    with psycopg.connect(migr_admin, autocommit=True) as c:
        assert c.execute("SELECT relkind FROM pg_class WHERE oid = to_regclass('retrieval_audit')").fetchone()[0] == "p"
        assert count(c) == 2 and count(c, "retrieval_audit_legacy") == 2  # legacy kept until the operator drops it
        months = [m for _, m in retention.list_partitions(c, "retrieval_audit")]
        assert months_ago(3).replace(day=1, hour=0, minute=0, second=0, microsecond=0) in months

    with tenant_db(migr_admin, migr_app) as db:  # the migrated table is usable under RLS
        with db.tenant_transaction("acme") as conn:
            assert conn.execute("SELECT count(*) FROM retrieval_audit").fetchone()[0] == 2
        with db.tenant_transaction("rival") as conn:
            assert conn.execute("SELECT count(*) FROM retrieval_audit").fetchone()[0] == 0
    admin.execute("DROP DATABASE migr_test WITH (FORCE)")


# ---- CLI ----------------------------------------------------------------------------------

def test_cli_retention_dry_run_reports_json_and_changes_nothing(pg_dsn, pg_app_dsn, admin):
    with tenant_db(pg_dsn, pg_app_dsn):
        admin.execute("DELETE FROM retrieval_audit")
        insert_audit(admin, months_ago(4))
        env = {**os.environ, "POSTGRES_ADMIN_DSN": pg_dsn}
        out = subprocess.run([sys.executable, "-m", "contextrag", "retention", "--months", "2", "--dry-run"],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        [report] = [r for r in json.loads(out.stdout) if r["table"] == "retrieval_audit"]
        assert report["dry_run"] and report["rows_removed"] == 1
        assert count(admin) == 1


def test_cli_retention_requires_a_dsn(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("POSTGRES")}
    out = subprocess.run([sys.executable, "-m", "contextrag", "retention", "--months", "1"], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 2 and "POSTGRES_ADMIN_DSN" in out.stderr


# ---- concurrency --------------------------------------------------------------------------

def test_pool_burst_keeps_tenants_isolated_and_loses_no_writes(pg_dsn, pg_app_dsn, admin):
    admin.execute("DELETE FROM retrieval_audit")
    tenants = ["t0", "t1", "t2", "t3"]

    # 32 workers share one pool of 4 connections: most transactions wait for a connection
    with tenant_db(pg_dsn, pg_app_dsn, max_size=4, timeout=30) as db:

        def work(i):
            tenant = tenants[i % len(tenants)]
            leaked = []
            for n in range(10):
                with db.tenant_transaction(tenant) as conn:
                    db.log_retrieval(conn, {"id": str(uuid.uuid4()), "tenant": tenant, "user": None,
                                            "query": f"{tenant}-{i}-{n}", "top_k": 1, "fragments": 0,
                                            "doc_ids": [], "latency_ms": 0.1})
                    seen = {t for (t,) in conn.execute("SELECT DISTINCT tenant FROM retrieval_audit").fetchall()}
                    if not seen <= {tenant}:
                        leaked.append(seen)
            return leaked

        with ThreadPoolExecutor(max_workers=16) as pool:
            leaks = [x for result in pool.map(work, range(32)) for x in result]
    assert leaks == []  # no transaction ever saw another tenant's rows
    expected = {t: 0 for t in tenants}
    for i in range(32):
        expected[tenants[i % len(tenants)]] += 10
    for t, n in expected.items():
        assert count(admin, where="tenant = %s", params=(t,)) == n


def test_a_pool_connection_never_carries_a_tenant_to_the_next_request(pg_dsn, pg_app_dsn):
    with tenant_db(pg_dsn, pg_app_dsn, max_size=1) as db:
        with db.tenant_transaction("acme") as conn:
            assert conn.execute("SELECT current_setting('app.tenant', true)").fetchone()[0] == "acme"
        with db.pool.connection() as conn:  # the same single connection, outside any tenant transaction
            assert not conn.execute("SELECT nullif(current_setting('app.tenant', true), '')").fetchone()[0]


def test_pool_exhaustion_fails_closed_with_503_then_recovers(pg_dsn, pg_app_dsn, admin):
    rag = factory.offline()
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    with tenant_db(pg_dsn, pg_app_dsn, max_size=1, timeout=0.3) as db:
        client = TestClient(create_app(rag, tenant_db=db))
        body = {"tenant_id": "acme", "query": QUESTION}
        with db.tenant_transaction("acme"):  # hold the only connection, as a slow request would
            busy = client.post("/api/v1/retrieve", json=body, headers={"X-Tenant": "acme"})
        assert busy.status_code == 503 and busy.json()["error"] == "database_unavailable"
        assert busy.headers["Retry-After"] == "5"
        ok = client.post("/api/v1/retrieve", json=body, headers={"X-Tenant": "acme"})
        assert ok.status_code == 200 and ok.json()["status"] == "ok"
        assert 'contextrag_retrieve_failed_total{reason="database_unavailable"} 1' in rag.metrics_text()


def test_api_burst_through_a_small_pool_serves_every_request(pg_dsn, pg_app_dsn, admin):
    admin.execute("DELETE FROM retrieval_audit")
    rag = factory.offline()
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    body = {"tenant_id": "acme", "query": QUESTION}
    with tenant_db(pg_dsn, pg_app_dsn, max_size=3, timeout=30) as db:
        client = TestClient(create_app(rag, tenant_db=db))

        def hit(_):
            return client.post("/api/v1/retrieve", json=body, headers={"X-Tenant": "acme"}).status_code

        with ThreadPoolExecutor(max_workers=12) as pool:
            statuses = list(pool.map(hit, range(60)))
    assert statuses == [200] * 60
    assert count(admin, where="tenant = 'acme'") == 60  # every served request was audited, none were lost
