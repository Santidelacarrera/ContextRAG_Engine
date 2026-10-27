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

import asyncio
import binascii
import hashlib
import hmac
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..core.security import visible
from ..core.config import Settings
from ..core.redact import install_log_redaction
from ..core.tracing import request_id_var
from ..core.ingestion import DocumentRejected, EmptyDocumentError, from_text, load_base64
from ..generation.llm import GenerationError
from ..pipeline import ContextRAG, ConversationUnavailable
from .auth import KeyStore
from .ratelimit import ConcurrencyGate, RateLimiter
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
    # An admin whose key carries groups manages only those cases (groups); an admin without groups manages the tenant.
    restricted: bool = False


def scoped_conversation_id(owner: str | None, conversation_id: str) -> str:
    """The id a conversation is stored under. Binding it to its owner makes a guessed or leaked client-side id useless to
    anyone else in the same tenant: another user sending the same id gets a different, empty conversation."""
    return hashlib.sha256(f"{owner or ''}\0{conversation_id}".encode()).hexdigest()


def may_manage(caller: Caller, acl: tuple[str, ...] | list[str] | None) -> bool:
    """May ``caller`` create, replace or delete a document with this ACL? ``None`` means the document does not exist.
    Unrestricted callers may manage anything in their tenant. A case-restricted admin may manage a document only when
    it is bound to at least one case and every case it is bound to belongs to the caller."""
    if not caller.restricted:
        return True
    return bool(acl) and set(acl or ()) <= set(caller.groups or ())


def create_app(rag: ContextRAG | None = None, keystore: KeyStore | None = None,
               rate_limit_per_min: int | None = None, tenant_db=None) -> FastAPI:
    install_log_redaction()  # secrets and common identifiers never reach a log line, whatever the caller logged
    state = {"rag": rag, "db": tenant_db}
    owned = {"rag": False, "db": False}  # only what this app built from the environment is closed on shutdown

    @asynccontextmanager
    async def lifespan(_app):
        yield
        # Graceful shutdown: stop the compaction worker (running attempts finish, queued ones stay queued for the next
        # start) and close the stores. close() blocks, so it runs off the event loop.
        for key in ("rag", "db"):
            if owned[key] and state[key] is not None:
                await asyncio.to_thread(state[key].close)

    app = FastAPI(title="ContextRAG Engine", version="0.3.0", lifespan=lifespan)
    keystore = keystore if keystore is not None else KeyStore.from_env()
    if keystore is None and os.getenv("CONTEXTRAG_REQUIRE_AUTH", "").strip().lower() in ("1", "true", "yes", "on"):
        # Open and shared-secret modes trust the tenant and groups the caller claims. For any multi-user deployment that
        # is a data leak by design, so the operator can make the server refuse to run in those modes.
        raise RuntimeError("CONTEXTRAG_REQUIRE_AUTH is set but no key store is configured: set CONTEXTRAG_KEYS_FILE "
                           "(API keys bound to a tenant, cases and a role); open and shared-secret modes are refused")
    if rate_limit_per_min is None and os.getenv("CONTEXTRAG_RATE_LIMIT_PER_MIN"):
        rate_limit_per_min = int(os.environ["CONTEXTRAG_RATE_LIMIT_PER_MIN"])
    limiter = RateLimiter(rate_limit_per_min) if rate_limit_per_min else None
    cfg = getattr(rag, "cfg", None) or Settings.from_env()  # an injected pipeline brings its own settings
    query_gate = ConcurrencyGate(cfg.max_concurrent_queries, cfg.busy_wait_seconds)
    ingest_gate = ConcurrencyGate(cfg.max_concurrent_ingests, cfg.busy_wait_seconds)

    def gated(gate: ConcurrencyGate, kind: str):
        def dependency():
            if not gate.acquire():
                if state["rag"] is not None:
                    state["rag"].count(f'requests_rejected_busy_total{{endpoint="{kind}"}}')
                raise HTTPException(503, "the server is busy; retry shortly", headers={"Retry-After": "2"})
            try:
                yield
            finally:
                gate.release()

        return dependency

    @app.middleware("http")
    async def request_id_header(request, call_next):
        """Every response carries X-Request-Id: a well-formed inbound id is kept for correlation, anything else is
        replaced by a fresh UUID. Errors raised before a handler runs (401, 403, 422) get one too."""
        inbound = request.headers.get("x-request-id", "")
        request_id = inbound if REQUEST_ID.fullmatch(inbound) else str(uuid.uuid4())
        request.state.request_id = request_id
        request_id_var.set(request_id)  # lets an exported trace carry the same id the client and the logs see
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    if keystore is None and not os.getenv("CONTEXTRAG_API_KEY"):
        log.warning("API is running without authentication (set CONTEXTRAG_KEYS_FILE or CONTEXTRAG_API_KEY)")

    @app.exception_handler(GenerationError)
    async def generation_unavailable(request, exc):
        log.error("generation failed: %s request_id=%s", type(exc.__cause__ or exc).__name__,
                  getattr(request.state, "request_id", "-"))  # the provider's message may echo request details
        return JSONResponse({"error": "generation_unavailable",
                             "detail": "the language model could not be reached; no answer was produced"},
                            status_code=503, headers={"Retry-After": "5"})

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        """Anything not handled above: a fixed body with the request id. The exception text can carry paths, DSNs or
        document content, so it is never returned; the class and request id are logged for correlation."""
        request_id = getattr(request.state, "request_id", "-")
        log.error("unhandled %s request_id=%s", type(exc).__name__, request_id,
                  exc_info=bool(os.getenv("CONTEXTRAG_DEBUG_ERRORS")))
        return JSONResponse({"error": "internal_error", "request_id": request_id}, status_code=500)

    def get_rag() -> ContextRAG:
        if state["rag"] is None:
            from .. import factory

            state["rag"] = factory.from_env()
            owned["rag"] = True
        return state["rag"]

    def get_db():
        if state["db"] is None:
            from ..db import TenantDB

            state["db"] = TenantDB.from_env()  # None when POSTGRES_DSN is not set
            owned["db"] = state["db"] is not None
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
                caller = Caller(p.id, p.tenant, p.groups, restricted=(p.role == "admin" and bool(p.groups)))
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

    @app.get("/me")
    def me(x_api_key: str | None = Header(None)):
        """Who the presented key is: its id (the audit identifier), tenant, role and groups. It reveals nothing the key
        does not already grant, and lets a consumer check, before acting, that a key belongs to the organisation it
        expects. Needs a valid key; with no key store configured there is no identity to report."""
        if keystore is None:
            raise HTTPException(404, "identity is not available without a key store")
        p = keystore.authenticate(x_api_key)
        if p is None:
            raise HTTPException(401, "invalid or missing X-API-Key")
        return {"key_id": p.id, "tenant": p.tenant, "role": p.role, "groups": sorted(p.groups)}

    @app.get("/healthz")
    def healthz():
        try:
            parts = get_rag().health()
        except Exception as e:  # backend unreachable at startup
            # Public endpoint: the class of the failure only. A connection error names hosts, ports and users.
            log.warning("health check failed: %s", type(e).__name__)
            return JSONResponse({"status": "unavailable", "error": type(e).__name__}, status_code=503)
        ok = all(parts.values())
        return JSONResponse({"status": "ok" if ok else "degraded", "components": parts}, status_code=200 if ok else 503)

    @app.post("/ingest", dependencies=[Depends(gated(ingest_gate, "ingest"))])
    def ingest(body: IngestIn, c: Caller = Depends(caller_for("ingest"))):
        rag = get_rag()
        results = []
        for d in body.documents:
            if (d.text is None) == (d.content_base64 is None):
                raise HTTPException(422, f"{d.doc_id}: provide exactly one of text / content_base64")
            name = d.source or d.doc_id
            if c.restricted:  # a case admin may only write into their own cases, and never over another case's document
                if not may_manage(c, d.acl) or not may_manage(c, rag.index.doc_acl(c.tenant, d.doc_id) or d.acl):
                    raise HTTPException(403, f"{d.doc_id}: not permitted for this key's cases")
            try:
                doc = (from_text(d.text, name, d.kind) if d.text is not None
                       else load_base64(d.content_base64 or "", name, rag.cfg))
                if d.kind and d.text is None and doc.kind != "pdf":
                    doc.kind = d.kind
            except DocumentRejected as e:  # a stable code and no parser text; oversize is 413, the rest is 422
                raise HTTPException(413 if e.code == "too_large" else 422, f"{d.doc_id}: {e}")
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

    @app.post("/query", dependencies=[Depends(gated(query_gate, "query"))])
    def query(body: QueryIn, c: Caller = Depends(caller_for("query")), x_user: str | None = Header(None)):
        if c.groups is None:  # unbound modes: trust the request
            groups = frozenset(body.groups)
        else:  # bound to a key: a request may only narrow the key's groups, never widen them
            extra = set(body.groups) - c.groups
            if extra:
                raise HTTPException(403, f"groups not granted to this key: {sorted(extra)}")
            groups = frozenset(body.groups) if body.groups else c.groups
        try:
            owner = c.id or x_user
            a = get_rag().ask(body.question, tenant=c.tenant, groups=groups, user=owner, top_k=body.top_k,
                              conversation_id=(scoped_conversation_id(owner, body.conversation_id)
                                               if body.conversation_id else None))
        except ConversationUnavailable:
            return JSONResponse({"error": "conversations_unavailable",
                                 "detail": "conversation memory is not configured on this server"}, status_code=503)
        return a.to_response()

    @app.delete("/conversations/{conversation_id}")
    def erase_conversation(conversation_id: str, c: Caller = Depends(caller_for("erase")),
                           x_user: str | None = Header(None)):
        """Erase one conversation of the caller: its turns, memories and compaction jobs. Idempotent."""
        rag = get_rag()
        if rag.memory is None:
            raise HTTPException(404, "conversation memory not configured")
        rag.memory.delete_conversation(c.tenant, scoped_conversation_id(c.id or x_user, conversation_id))
        return {"tenant": c.tenant, "deleted": conversation_id}

    register_retrieve(app, get_rag=get_rag, get_db=get_db, caller_dep=caller_for)

    @app.delete("/documents/{doc_id:path}")
    def delete_document(doc_id: str, c: Caller = Depends(caller_for("delete"))):
        """Remove a document (all its chunks, vectors and BM25 entries) for the tenant. Idempotent for tenant-wide
        admins. A case admin gets the same 404 for a document that is absent and one that belongs to another case."""
        rag = get_rag()
        if c.restricted and not may_manage(c, rag.index.doc_acl(c.tenant, doc_id)):
            raise HTTPException(404, "document not found")
        forgotten = rag.delete_document(c.tenant, doc_id)
        return {"tenant": c.tenant, "deleted": doc_id, "conversation_turns_removed": forgotten.get("turns", 0)}

    @app.delete("/tenant")
    def erase_tenant(c: Caller = Depends(caller_for("erase")), x_confirm_tenant: str | None = Header(None),
                     keep_audit: bool = False):
        """Irreversibly erase everything stored for the caller's tenant. Requires ``X-Confirm-Tenant: <tenant>``
        so that a stray DELETE cannot wipe a tenant. ``keep_audit=true`` preserves audit records (legal hold)."""
        if c.restricted:
            raise HTTPException(403, "erasing a tenant needs a key that is not restricted to cases")
        if x_confirm_tenant != c.tenant:
            raise HTTPException(400, "send header X-Confirm-Tenant with the exact tenant name to confirm erasure")
        return {"tenant": c.tenant, **get_rag().erase_tenant(c.tenant, include_audit=not keep_audit)}

    @app.get("/documents")
    def list_documents(c: Caller = Depends(caller_for("list"))):
        """Indexed documents of the tenant (including ACL-restricted ones: this is an admin view)."""
        docs = get_rag().index.list_documents(c.tenant)
        if c.restricted:  # only what this key could read anyway
            docs = [d for d in docs if visible(tuple(d.get("acl", ())), c.groups or frozenset())]
        return {"tenant": c.tenant, "documents": docs}

    @app.get("/metrics")
    def metrics(_: Caller = Depends(caller_for("metrics"))):
        return PlainTextResponse(get_rag().metrics_text(), media_type="text/plain; version=0.0.4")

    @app.get("/audit")
    def audit(limit: int = 50, c: Caller = Depends(caller_for("audit"))):
        log_ = get_rag().audit
        if log_ is None:
            raise HTTPException(404, "audit log not configured")
        limit = max(1, min(limit, 500))
        if c.restricted:  # a case admin sees their own queries, never other users' questions and answers
            return [r for r in log_.list(c.tenant, 500) if r.get("user") == c.id][:limit]
        return log_.list(c.tenant, limit)

    return app


def app_factory() -> FastAPI:
    """For ``uvicorn --factory contextrag.api.server:app_factory``."""
    return create_app()
