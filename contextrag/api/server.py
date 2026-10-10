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


MAX_TEXT_CHARS = 5_000_000
MAX_B64_CHARS = 30_000_000  # ~22 MB of raw bytes
MAX_DOCS_PER_REQUEST = 50


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=512)
    text: str | None = Field(None, max_length=MAX_TEXT_CHARS, description="Plain text / markdown / source code")
    content_base64: str | None = Field(None, max_length=MAX_B64_CHARS,
                                       description="Raw file bytes (e.g. a PDF), base64-encoded")
    source: str | None = Field(None, max_length=512,
                               description="Display name; its extension selects the parser (.md/.py/.pdf...)")
    kind: str | None = Field(None, pattern="^(text|markdown|code)$")
    acl: list[str] = Field([], max_length=64)


class IngestIn(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=MAX_DOCS_PER_REQUEST)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    groups: list[str] = Field([], max_length=64)
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
        return a.to_response()

    @app.delete("/documents/{doc_id:path}", dependencies=[Depends(auth)])
    def delete_document(doc_id: str, t: str = Depends(tenant)):
        """Remove a document (all its chunks, vectors and BM25 entries) for the tenant. Idempotent."""
        get_rag().ingestor.delete(tenant=t, doc_id=doc_id)
        return {"tenant": t, "deleted": doc_id}

    @app.get("/metrics", dependencies=[Depends(auth)])
    def metrics():
        from fastapi.responses import PlainTextResponse

        return PlainTextResponse(get_rag().metrics_text(), media_type="text/plain; version=0.0.4")

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
