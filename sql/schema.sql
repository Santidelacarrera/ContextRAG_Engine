CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS answer_audit (
    id              uuid PRIMARY KEY,
    ts              timestamptz NOT NULL DEFAULT now(),
    tenant          text NOT NULL,
    user_id         text,
    question        text NOT NULL,
    status          text NOT NULL CHECK (status IN ('verified', 'fallback')),
    answer          text NOT NULL,
    confidence      double precision NOT NULL,
    attempts        integer NOT NULL,
    reason          text NOT NULL DEFAULT '',
    claims          jsonb NOT NULL,   -- per-claim verdict + evidence coordinates
    sources         jsonb NOT NULL,   -- parent chunks supplied to the LLM
    query_embedding vector(1024)      -- adjust to the embedder's dimension
);
CREATE INDEX IF NOT EXISTS answer_audit_tenant_ts ON answer_audit (tenant, ts DESC);
