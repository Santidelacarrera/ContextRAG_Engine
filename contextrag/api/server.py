"""FastAPI service.

Tenant is taken from the ``X-Tenant`` header. If ``CONTEXTRAG_API_KEY`` is set, every call except
``/healthz`` must also send a matching ``X-API-Key``; real authentication/tenant binding should still
be enforced by a gateway in front of this service.
"""
from __future__ import annotations

import binascii
import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..core.ingestion import LoadedDocument, load_base64
from ..pipeline import ContextRAG


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1)
    text: str | None = Field(None, description="Plain text / markdown / source code")
    content_base64: str | None = Field(None, description="Raw file bytes (e.g. a PDF), base64-encoded")
    source: str | None = Field(None, description="Display name; its extension selects the parser (.md/.py/.pdf...)")
    kind: str | None = Field(None, pattern="^(text|markdown|code)$")
    acl: list[str] = []


class IngestIn(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1)


class QueryIn(BaseModel):
    question: str = Field(min_length=1)
    groups: list[str] = []
    top_k: int | None = Field(None, ge=1, le=20)


def create_app(rag: ContextRAG | None = None) -> FastAPI:
    app = FastAPI(title="ContextRAG Engine", version="0.2.0")
    state = {"rag": rag}

    def get_rag() -> ContextRAG:
        if state["rag"] is None:
            from .. import factory

            state["rag"] = factory.from_env()
        return state["rag"]

    def auth(x_api_key: str | None = Header(None)):
        expected = os.getenv("CONTEXTRAG_API_KEY")
        if expected and not hmac.compare_digest(x_api_key or "", expected):
            raise HTTPException(401, "invalid or missing X-API-Key")

    def tenant(x_tenant: str | None = Header(None)) -> str:
        if not x_tenant or not x_tenant.strip():
            raise HTTPException(400, "X-Tenant header required")
        return x_tenant.strip()

    @app.get("/healthz")
    def healthz():
        from fastapi.responses import JSONResponse

        try:
            parts = get_rag().health()
        except Exception as e:  # backend unreachable at startup
            return JSONResponse({"status": "unavailable", "error": str(e)}, status_code=503)
        ok = all(parts.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "components": parts}, status_code=200 if ok else 503)

    @app.post("/ingest", dependencies=[Depends(auth)])
    def ingest(body: IngestIn, t: str = Depends(tenant)):
        rag = get_rag()
        results = []
        for d in body.documents:
            if (d.text is None) == (d.content_base64 is None):
                raise HTTPException(422, f"{d.doc_id}: provide exactly one of text / content_base64")
            name = d.source or d.doc_id
            try:
                doc = (LoadedDocument(d.text, d.kind or _kind(name)) if d.text is not None
                       else load_base64(d.content_base64, name))
                if d.kind and doc.kind != "pdf":
                    doc.kind = d.kind
            except (binascii.Error, ValueError) as e:
                raise HTTPException(422, f"{d.doc_id}: invalid content ({e})")
            except Exception as e:  # unreadable PDF etc.
                raise HTTPException(422, f"{d.doc_id}: could not parse document ({type(e).__name__})")
            results.append(rag.ingestor.ingest(doc, tenant=t, doc_id=d.doc_id, source=name, acl=tuple(d.acl)))
        return {"tenant": t, "documents": results}

    @app.post("/query", dependencies=[Depends(auth)])
    def query(body: QueryIn, t: str = Depends(tenant), x_user: str | None = Header(None)):
        a = get_rag().ask(body.question, tenant=t, groups=frozenset(body.groups), user=x_user, top_k=body.top_k)
        return {
            "status": a.status.value,
            "answer": a.text,
            "confidence": a.confidence,
            "attempts": a.attempts,
            "reason": a.reason,
            "cached": a.cached,
            "claims": [{"claim": c.claim, "citations": c.citations, "score": c.score, "supported": c.supported,
                        "reason": c.reason,
                        "evidence": None if c.evidence is None else {
                            "doc_id": c.evidence.doc_id, "source": c.evidence.source, "start": c.evidence.start,
                            "end": c.evidence.end, "page": c.evidence.page, "text": c.evidence.text}}
                       for c in a.claims],
            "sources": [{"index": s.index, "doc_id": s.parent.doc_id, "source": s.parent.source,
                         "start": s.parent.start, "end": s.parent.end, "page": s.parent.page,
                         "section": s.parent.section, "score": s.score} for s in a.sources],
        }

    @app.get("/audit", dependencies=[Depends(auth)])
    def audit(limit: int = 50, t: str = Depends(tenant)):
        log = get_rag().audit
        if log is None:
            raise HTTPException(404, "audit log not configured")
        return log.list(t, max(1, min(limit, 500)))

    return app


def _kind(name: str) -> str:
    from ..core.chunking import detect_kind

    k = detect_kind(name)
    return "text" if k == "pdf" else k


def app_factory() -> FastAPI:
    """For ``uvicorn --factory contextrag.api.server:app_factory``."""
    return create_app()
