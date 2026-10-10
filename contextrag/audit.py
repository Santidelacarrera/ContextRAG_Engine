"""Audit trail of every query: question, verdict, per-claim evidence coordinates and sources."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Protocol

from .core.models import Answer


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

    def health(self) -> bool: ...


class InMemoryAudit:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, record, query_embedding=None):
        self.records.append(record)

    def list(self, tenant, limit=50):
        return [r for r in reversed(self.records) if r["tenant"] == tenant][:limit]

    def health(self):
        return True


class PostgresAudit:
    """PostgreSQL audit table. If the ``vector`` extension is available the query embedding is
    stored as ``vector(dim)`` (pgvector, enabling similar-query lookups); otherwise as ``float8[]``."""

    def __init__(self, dsn: str, dim: int):
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)
        try:
            self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self.has_vector = True
        except Exception:
            self.has_vector = False
        col = f"vector({int(dim)})" if self.has_vector else "double precision[]"
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS answer_audit (
                id uuid PRIMARY KEY,
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
                query_embedding {col}
            )""")
        self.conn.execute("CREATE INDEX IF NOT EXISTS answer_audit_tenant_ts ON answer_audit (tenant, ts DESC)")

    def write(self, record, query_embedding=None):
        emb = None
        if query_embedding is not None:
            emb = "[" + ",".join(map(str, query_embedding)) + "]" if self.has_vector else list(query_embedding)
        cast = "%s::vector" if self.has_vector else "%s"
        self.conn.execute(
            f"""INSERT INTO answer_audit (id, ts, tenant, user_id, question, status, answer, confidence,
                    attempts, reason, cached, claims, sources, query_embedding)
                VALUES (%s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, {cast})""",
            (record["id"], record["ts"], record["tenant"], record["user"], record["question"], record["status"],
             record["answer"], record["confidence"], record["attempts"], record["reason"], record["cached"],
             json.dumps(record["claims"]), json.dumps(record["sources"]), emb))

    def list(self, tenant, limit=50):
        cur = self.conn.execute(
            """SELECT id, extract(epoch from ts), user_id, question, status, answer, confidence, attempts, reason,
                      cached, claims, sources
               FROM answer_audit WHERE tenant = %s ORDER BY ts DESC LIMIT %s""", (tenant, limit))
        keys = ("id", "ts", "user", "question", "status", "answer", "confidence", "attempts", "reason",
                "cached", "claims", "sources")
        return [{**dict(zip(keys, r, strict=True)), "id": str(r[0]), "tenant": tenant} for r in cur.fetchall()]

    def health(self):
        try:
            self.conn.execute("SELECT 1")
            return True
        except Exception:
            return False
