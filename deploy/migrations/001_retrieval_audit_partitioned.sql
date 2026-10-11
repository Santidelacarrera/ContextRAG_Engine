-- Migration 001: retrieval_audit becomes RANGE-partitioned by month (UTC) on ts.
--
-- Needed once, on databases created before partitioning (plain retrieval_audit). New installs get the partitioned
-- table from the application itself, so there is nothing to run there.
--
--   psql "$POSTGRES_ADMIN_DSN" -v ON_ERROR_STOP=1 -v app_role=contextrag_app \
--        -f deploy/migrations/001_retrieval_audit_partitioned.sql
--
-- Runs as the table owner, in one transaction. The old table is renamed to retrieval_audit_legacy and kept:
-- check the row counts, then drop it yourself:  DROP TABLE retrieval_audit_legacy;
-- Running it twice fails safely (the guard below refuses to touch a table that is not a plain table).

SET TIME ZONE 'UTC';
BEGIN;

DO $$
BEGIN
    IF (SELECT relkind FROM pg_class WHERE oid = to_regclass('retrieval_audit')) IS DISTINCT FROM 'r' THEN
        RAISE EXCEPTION 'retrieval_audit is not a plain table: nothing to migrate';
    END IF;
END $$;

LOCK TABLE retrieval_audit IN ACCESS EXCLUSIVE MODE;
ALTER TABLE retrieval_audit RENAME TO retrieval_audit_legacy;
ALTER INDEX IF EXISTS retrieval_audit_tenant_ts RENAME TO retrieval_audit_legacy_tenant_ts;

CREATE TABLE retrieval_audit (
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

-- one partition per month that holds legacy rows, plus the current and the next month
DO $$
DECLARE
    m timestamptz;
BEGIN
    FOR m IN
        SELECT DISTINCT date_trunc('month', ts) FROM retrieval_audit_legacy
        UNION SELECT date_trunc('month', now())
        UNION SELECT date_trunc('month', now()) + interval '1 month'
    LOOP
        EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF retrieval_audit FOR VALUES FROM (%L) TO (%L)',
                       'retrieval_audit_' || to_char(m, 'YYYYMM'), m, m + interval '1 month');
    END LOOP;
END $$;

INSERT INTO retrieval_audit SELECT * FROM retrieval_audit_legacy;

CREATE INDEX retrieval_audit_tenant_ts ON retrieval_audit (tenant, ts DESC);
ALTER TABLE retrieval_audit ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON retrieval_audit
    USING (tenant = current_setting('app.tenant', true))
    WITH CHECK (tenant = current_setting('app.tenant', true));
GRANT SELECT, INSERT ON retrieval_audit TO :"app_role";

COMMIT;
