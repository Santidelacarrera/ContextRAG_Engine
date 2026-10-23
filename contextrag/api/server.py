"""FastAPI service.

Authentication modes (first match wins):

1. **Key store** (``CONTEXTRAG_KEYS_FILE``): each ``X-API-Key`` is bound to a tenant, ACL groups and a role
   (see ``contextrag.api.auth``). The tenant/groups are taken from the key; a conflicting ``X-Tenant`` is
   refused (403) and request ``groups`` may only *narrow* the key's groups. Use this in production.
2. **Shared secret** (``CONTEXTRAG_API_KEY``): one key for everybody; ``X-Tenant`` and ``groups`` are trusted
   as sent, so a gateway must set them.
3. **Open** (development): tenant from ``X-Tenant``, everything trusted.

``/healthz`` is always open. ``CONTEXTRAG_RATE_LIMIT_PER_MIN`` limits calls per tenant (HTTP 429).
"""
from __future__ import annotations

import binascii
import hmac
import logging
import os
import re
import uuid
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..core.ingestion import EmptyDocumentError, from_text, load_base64
from ..generation.llm import GenerationError
from ..pipeline import ContextRAG, ConversationUnavailable
from .auth import KeyStore
from .ratelimit import RateLimiter
from .retrieval import register_retrieve

log = logging.getLogger("contextrag")

MAX_TEXT_CHARS = 5_000_000
MAX_B64_CHARS = 30_000_000  # ~22 MB of raw bytes
MAX_DOCS_PER_REQUEST = 50
REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class DocumentIn(BaseModel):
    doc_id: str = Field(min_length=1, max_length=512)
    text: str | None = Field(None, max_length=MAX_TEXT_CHARS, description="Plain text / markdown / source code")
    content_base64: str | None = Field(None, max_length=MAX_B64_CHARS,
                                       description="Raw file bytes (e.g. a PDF), base64-encoded")
    source: str | None = Field(None, max_length=512,
                               description="Display name; its extension selects the parser (.md/.py/.pdf...)")
    kind: str | None = Field(None, pattern="^(text|markdown|code)$")
    acl: list[str] = Field([], max_length=64)
    force: bool = Field(False, description="Re-embed even if the document is unchanged")


class IngestIn(BaseModel):
    documents: list[DocumentIn] = Field(min_length=1, max_length=MAX_DOCS_PER_REQUEST)


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    groups: list[str] = Field([], max_length=64)
    top_k: int | None = Field(None, ge=1, le=20)
    conversation_id: str | None = Field(None, pattern=r"^[A-Za-z0-9._:-]{1,128}$",
                                        description="Continue a conversation: history and memories are used, "
                                                    "and the turn is stored. Requires MEMORY_DB.")


@dataclass(frozen=True)
class Caller:
    id: str | None
    tenant: str
    groups: frozenset[str] | None  # None => unbound: the request body's groups are trusted


def create_app(rag: ContextRAG | None = None, keystore: KeyStore | None = None,
               rate_limit_per_min: int | None = None, tenant_db=None) -> FastAPI:
    app = FastAPI(title="ContextRAG Engine", version="0.3.0")
    state = {"rag": rag, "db": tenant_db}
    keystore = keystore if keystore is not None else KeyStore.from_env()
    if rate_limit_per_min is None and os.getenv("CONTEXTRAG_RATE_LIMIT_PER_MIN"):
        rate_limit_per_min = int(os.environ["CONTEXTRAG_RATE_LIMIT_PER_MIN"])
    limiter = RateLimiter(rate_limit_per_min) if rate_limit_per_min else None

    @app.middleware("http")
    async def request_id_header(request, call_next):
        """Every response carries X-Request-Id: a well-formed inbound id is kept for correlation, anything else is
        replaced by a fresh UUID. Errors raised before a handler runs (401, 403, 422) get one too."""
        inbound = request.headers.get("x-request-id", "")
        request_id = inbound if REQUEST_ID.fullmatch(inbound) else str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    if keystore is None and not os.getenv("CONTEXTRAG_API_KEY"):
        log.warning("API is running without authentication (set CONTEXTRAG_KEYS_FILE or CONTEXTRAG_API_KEY)")

    @app.exception_handler(GenerationError)
    async def generation_unavailable(request, exc):
        log.error("generation failed: %s", exc)
        return JSONResponse({"error": "generation_unavailable",
                             "detail": "the language model could not be reached; no answer was produced"},
                            status_code=503, headers={"Retry-After": "5"})

    def get_rag() -> ContextRAG:
        if state["rag"] is None:
            from .. import factory

            state["rag"] = factory.from_env()
        return state["rag"]

    def get_db():
        if state["db"] is None:
            from ..db import TenantDB

            state["db"] = TenantDB.from_env()  # None when POSTGRES_DSN is not set
        return state["db"]

    def caller_for(action: str):
        def dep(x_api_key: str | None = Header(None), x_tenant: str | None = Header(None),
                x_user: str | None = Header(None)) -> Caller:
            claimed = x_tenant.strip() if x_tenant and x_tenant.strip() else None
            if keystore is not None:
                p = keystore.authenticate(x_api_key)
                if p is None:
                    raise HTTPException(401, "invalid or missing X-API-Key")
                if not p.can(action):
                    raise HTTPException(403, f"role '{p.role}' may not {action}")
                if claimed and p.tenant and claimed != p.tenant:
                    raise HTTPException(403, "X-Tenant does not match the API key's tenant")
                caller = Caller(p.id, p.tenant, p.groups)
            else:
                expected = os.getenv("CONTEXTRAG_API_KEY")
                if expected and not hmac.compare_digest(x_api_key or "", expected):
                    raise HTTPException(401, "invalid or missing X-API-Key")
                if action == "metrics":
                    caller = Caller(x_user, claimed or "", None)
                else:
                    if not claimed:
                        raise HTTPException(400, "X-Tenant header required")
                    caller = Caller(x_user, claimed, None)
            if limiter:
                wait = limiter.acquire(caller.tenant or caller.id or "-")
                if wait > 0:
                    raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": str(int(wait) + 1)})
            return caller

        return dep

    @app.get("/healthz")
    def healthz():
        try:
            parts = get_rag().health()
        except Exception as e:  # backend unreachable at startup
            return JSONResponse({"status": "unavailable", "error": str(e)}, status_code=503)
        ok = all(parts.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "components": parts}, status_code=200 if ok else 503)

    @app.post("/ingest")
    def ingest(body: IngestIn, c: Caller = Depends(caller_for("ingest"))):
        rag = get_rag()
        results = []
        for d in body.documents:
            if (d.text is None) == (d.content_base64 is None):
                raise HTTPException(422, f"{d.doc_id}: provide exactly one of text / content_base64")
            name = d.source or d.doc_id
            try:
                doc = from_text(d.text, name, d.kind) if d.text is not None else load_base64(d.content_base64, name)
                if d.kind and d.text is None and doc.kind != "pdf":
                    doc.kind = d.kind
            except (binascii.Error, ValueError) as e:
                raise HTTPException(422, f"{d.doc_id}: invalid content ({e})")
            except Exception as e:  # unreadable PDF etc.
                raise HTTPException(422, f"{d.doc_id}: could not parse document ({type(e).__name__})")
            try:
                results.append(rag.ingestor.ingest(doc, tenant=c.tenant, doc_id=d.doc_id, source=name,
                                                   acl=tuple(d.acl), force=d.force))
            except EmptyDocumentError as e:
                raise HTTPException(422, str(e))
        return {"tenant": c.tenant, "documents": results}

    @app.post("/query")
    def query(body: QueryIn, c: Caller = Depends(caller_for("query")), x_user: str | None = Header(None)):
        if c.groups is None:  # unbound modes: trust the request
            groups = frozenset(body.groups)
        else:  # bound to a key: a request may only narrow the key's groups, never widen them
            extra = set(body.groups) - c.groups
            if extra:
                raise HTTPException(403, f"groups not granted to this key: {sorted(extra)}")
            groups = frozenset(body.groups) if body.groups else c.groups
        try:
            a = get_rag().ask(body.question, tenant=c.tenant, groups=groups, user=c.id or x_user, top_k=body.top_k,
                              conversation_id=body.conversation_id)
        except ConversationUnavailable:
            return JSONResponse({"error": "conversations_unavailable",
                                 "detail": "conversation memory is not configured on this server"}, status_code=503)
        return a.to_response()

    @app.delete("/conversations/{conversation_id}")
    def erase_conversation(conversation_id: str, c: Caller = Depends(caller_for("erase"))):
        """Erase one conversation of the caller's tenant: its turns, memories and compaction jobs. Idempotent."""
        rag = get_rag()
        if rag.memory is None:
            raise HTTPException(404, "conversation memory not configured")
        rag.memory.delete_conversation(c.tenant, conversation_id)
        return {"tenant": c.tenant, "deleted": conversation_id}

    register_retrieve(app, get_rag=get_rag, get_db=get_db, caller_dep=caller_for)

    @app.delete("/documents/{doc_id:path}")
    def delete_document(doc_id: str, c: Caller = Depends(caller_for("delete"))):
        """Remove a document (all its chunks, vectors and BM25 entries) for the tenant. Idempotent."""
        get_rag().ingestor.delete(tenant=c.tenant, doc_id=doc_id)
        return {"tenant": c.tenant, "deleted": doc_id}

    @app.delete("/tenant")
    def erase_tenant(c: Caller = Depends(caller_for("erase")), x_confirm_tenant: str | None = Header(None),
                     keep_audit: bool = False):
        """Irreversibly erase everything stored for the caller's tenant. Requires ``X-Confirm-Tenant: <tenant>``
        so that a stray DELETE cannot wipe a tenant. ``keep_audit=true`` preserves audit records (legal hold)."""
        if x_confirm_tenant != c.tenant:
            raise HTTPException(400, "send header X-Confirm-Tenant with the exact tenant name to confirm erasure")
        return {"tenant": c.tenant, **get_rag().erase_tenant(c.tenant, include_audit=not keep_audit)}

    @app.get("/documents")
    def list_documents(c: Caller = Depends(caller_for("list"))):
        """Indexed documents of the tenant (including ACL-restricted ones: this is an admin view)."""
        return {"tenant": c.tenant, "documents": get_rag().index.list_documents(c.tenant)}

    @app.get("/metrics")
    def metrics(_: Caller = Depends(caller_for("metrics"))):
        return PlainTextResponse(get_rag().metrics_text(), media_type="text/plain; version=0.0.4")

    @app.get("/audit")
    def audit(limit: int = 50, c: Caller = Depends(caller_for("audit"))):
        log_ = get_rag().audit
        if log_ is None:
            raise HTTPException(404, "audit log not configured")
        return log_.list(c.tenant, max(1, min(limit, 500)))

    return app


def app_factory() -> FastAPI:
    """For ``uvicorn --factory contextrag.api.server:app_factory``."""
    return create_app()
