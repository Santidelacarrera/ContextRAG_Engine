from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Protocol

from .models import Answer


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
        "claims": [asdict(c) for c in answer.claims],
        "sources": [
            {"index": c.index, "doc_id": c.parent.doc_id, "source": c.parent.source,
             "start": c.parent.start, "end": c.parent.end}
            for c in answer.sources
        ],
    }


class AuditLog(Protocol):
    def write(self, record: dict, query_embedding: list[float] | None = None) -> None: ...

    def list(self, tenant: str, limit: int = 50) -> list[dict]: ...


class InMemoryAudit:
    def __init__(self):
        self.records: list[dict] = []

    def write(self, record, query_embedding=None):
        self.records.append(record)

    def list(self, tenant, limit=50):
        return [r for r in reversed(self.records) if r["tenant"] == tenant][:limit]


class PostgresAudit:
    """Persists to ``sql/schema.sql`` (pgvector for similar-query lookup). Requires psycopg."""

    def __init__(self, dsn: str):
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)

    def write(self, record, query_embedding=None):
        emb = "[" + ",".join(map(str, query_embedding)) + "]" if query_embedding else None
        self.conn.execute(
            """INSERT INTO answer_audit (id, ts, tenant, user_id, question, status, answer, confidence,
                                         attempts, reason, claims, sources, query_embedding)
               VALUES (%s, to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::vector)""",
            (record["id"], record["ts"], record["tenant"], record["user"], record["question"], record["status"],
             record["answer"], record["confidence"], record["attempts"], record["reason"],
             json.dumps(record["claims"]), json.dumps(record["sources"]), emb),
        )

    def list(self, tenant, limit=50):
        cur = self.conn.execute(
            "SELECT id, extract(epoch from ts), question, status, answer, confidence, claims, sources "
            "FROM answer_audit WHERE tenant = %s ORDER BY ts DESC LIMIT %s", (tenant, limit))
        return [dict(id=str(r[0]), ts=r[1], tenant=tenant, question=r[2], status=r[3], answer=r[4],
                     confidence=r[5], claims=r[6], sources=r[7]) for r in cur.fetchall()]
