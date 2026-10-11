"""``POST /api/v1/retrieve``: hybrid retrieval for other services (e.g. an agent orchestrator).

Returns the matching fragments with their metadata and scores, plus a *preliminary* draft built from the
fragments by the offline extractive generator. The draft is not NLI-verified (``draft_verified`` is always false);
callers that need a verified answer should use ``POST /query``.

Tenant isolation is applied in two layers:

* the search itself filters by the caller's tenant (vector store and BM25 index);
* the retrieval audit record is written inside a transaction that sets ``app.tenant``, so the database's row-level
  security decides what the application role can read or write. The write is part of the request: if it fails,
  no fragments are returned (fail closed), because an access that cannot be audited must not succeed.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..generation.llm import NO_ANSWER, ExtractiveLLM
from ..pipeline import ContextRAG

log = logging.getLogger("contextrag.retrieve")

TENANT_PATTERN = r"^[A-Za-z0-9._:-]{1,128}$"
_extractive = ExtractiveLLM()


class RetrieveIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str = Field(pattern=TENANT_PATTERN, description="Must match the tenant bound to the caller")
    query: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=20)


class FragmentOut(BaseModel):
    index: int
    score: float
    doc_id: str
    source: str
    section: str
    page: int | None
    start: int
    end: int
    text: str


class RetrieveOut(BaseModel):
    status: Literal["ok", "no_results"]
    tenant_id: str
    query: str
    fragments: list[FragmentOut]
    draft_answer: str | None = Field(description="Preliminary, extractive and cited; not verified")
    draft_verified: Literal[False] = False
    latency_ms: float


def db_error_status(exc: Exception) -> tuple[int, str] | None:
    """HTTP status for a database failure, or None if ``exc`` is not one."""
    import psycopg
    from psycopg_pool import PoolClosed, PoolTimeout

    if isinstance(exc, psycopg.errors.InsufficientPrivilege):  # RLS denied the statement
        return 403, "tenant_not_authorized"
    if isinstance(exc, (psycopg.OperationalError, PoolTimeout, PoolClosed)):  # unreachable, exhausted or shut down
        return 503, "database_unavailable"
    if isinstance(exc, psycopg.Error):
        return 500, "database_error"
    return None


def register_retrieve(app: FastAPI, *, get_rag: Callable[[], ContextRAG], get_db: Callable[[], object | None],
                      caller_dep: Callable) -> None:
    @app.post("/api/v1/retrieve", response_model=RetrieveOut)
    def retrieve(body: RetrieveIn, c=Depends(caller_dep("query"))):
        if body.tenant_id != c.tenant:
            log.warning("retrieve refused tenant=%s claimed=%s reason=tenant_mismatch", c.tenant, body.tenant_id)
            raise HTTPException(403, "tenant_id does not match the caller's tenant")
        db = get_db()
        if db is None:
            log.error("retrieve unavailable tenant=%s reason=no_database", c.tenant)
            return JSONResponse({"error": "retrieval_store_unavailable",
                                 "detail": "POSTGRES_DSN is not configured"}, status_code=503)

        rag = get_rag()
        t0 = time.perf_counter()
        groups = c.groups or frozenset()  # unbound callers: public chunks only (no body-supplied groups here)
        try:
            with db.tenant_transaction(c.tenant) as conn:
                contexts = rag.retriever.retrieve(body.query, c.tenant, groups, body.top_k)
                latency_ms = (time.perf_counter() - t0) * 1000
                db.log_retrieval(conn, {
                    "id": str(uuid.uuid4()), "tenant": c.tenant, "user": c.id, "query": body.query,
                    "top_k": body.top_k, "fragments": len(contexts), "latency_ms": latency_ms,
                    "doc_ids": [ctx.parent.doc_id for ctx in contexts],
                })
        except Exception as exc:
            mapped = db_error_status(exc)
            if mapped is None:
                raise
            status, code = mapped
            log.error("retrieve failed tenant=%s user=%s error=%s status=%d", c.tenant, c.id, code, status,
                      exc_info=status == 500)
            headers = {"Retry-After": "5"} if status == 503 else None
            return JSONResponse({"error": code, "detail": "the request was not served; see server logs"},
                                status_code=status, headers=headers)

        fragments = [FragmentOut(index=ctx.index, score=ctx.score, doc_id=ctx.parent.doc_id,
                                 source=ctx.parent.source, section=ctx.parent.section, page=ctx.parent.page,
                                 start=ctx.parent.start, end=ctx.parent.end, text=ctx.parent.text)
                     for ctx in contexts]
        draft = _extractive.generate(body.query, contexts) if contexts else NO_ANSWER
        log.info("retrieve tenant=%s user=%s status=%s fragments=%d top_k=%d latency_ms=%.1f",
                 c.tenant, c.id, "ok" if fragments else "no_results", len(fragments), body.top_k, latency_ms)
        return RetrieveOut(status="ok" if fragments else "no_results", tenant_id=c.tenant, query=body.query,
                           fragments=fragments, draft_answer=None if draft == NO_ANSWER else draft,
                           latency_ms=round(latency_ms, 2))
