"""FastAPI service. Tenant comes from the ``X-Tenant`` header (put real authn in front of this)."""
from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from .pipeline import ContextRAG


class IngestIn(BaseModel):
    doc_id: str
    text: str
    source: str | None = None
    acl: list[str] = []


class AskIn(BaseModel):
    question: str
    groups: list[str] = []


def create_app(rag: ContextRAG | None = None) -> FastAPI:
    if rag is None:
        from . import factory

        url = os.getenv("QDRANT_URL")
        rag = factory.production(qdrant_url=url, postgres_dsn=os.getenv("POSTGRES_DSN")) if url else factory.offline()
    app = FastAPI(title="ContextRAG Engine")

    def tenant_of(x_tenant: str | None) -> str:
        if not x_tenant:
            raise HTTPException(400, "X-Tenant header required")
        return x_tenant

    @app.post("/ingest")
    def ingest(body: IngestIn, x_tenant: str | None = Header(None)):
        n = rag.ingestor.ingest_text(body.text, tenant=tenant_of(x_tenant), doc_id=body.doc_id,
                                     source=body.source, acl=tuple(body.acl))
        return {"doc_id": body.doc_id, "chunks": n}

    @app.post("/ask")
    def ask(body: AskIn, x_tenant: str | None = Header(None), x_user: str | None = Header(None)):
        a = rag.ask(body.question, tenant=tenant_of(x_tenant), groups=frozenset(body.groups), user=x_user)
        return {
            "status": a.status.value, "answer": a.text, "confidence": a.confidence, "attempts": a.attempts,
            "reason": a.reason,
            "claims": [asdict(c) for c in a.claims],
            "sources": [{"index": c.index, "doc_id": c.parent.doc_id, "source": c.parent.source,
                         "start": c.parent.start, "end": c.parent.end} for c in a.sources],
        }

    @app.get("/audit")
    def audit(limit: int = 50, x_tenant: str | None = Header(None)):
        if not rag.audit:
            raise HTTPException(404, "audit log not configured")
        return rag.audit.list(tenant_of(x_tenant), limit)

    return app
