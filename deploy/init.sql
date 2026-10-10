-- Runs once on first start of the postgres container. The audit table itself is created by
-- contextrag.audit.PostgresAudit (it needs the embedding dimension).
CREATE EXTENSION IF NOT EXISTS vector;
