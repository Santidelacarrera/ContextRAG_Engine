"""Local sparse retrieval with ``rank_bm25`` (BM25Plus), kept per tenant.

BM25Plus is used instead of BM25Okapi because Okapi's IDF turns negative when a term
occurs in more than half of the documents, which breaks small corpora. The index is
immutable in rank_bm25, so it is rebuilt lazily after any change.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Iterable

from rank_bm25 import BM25Plus

from ..core.models import ChildChunk
from ..core.security import visible
from ..core.text import content_tokens, stem


def terms(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


@dataclass
class _TenantIndex:
    chunks: list[ChildChunk]
    termsets: list[frozenset[str]]
    bm25: BM25Plus | None


class BM25Index:
    def __init__(self, loader: Callable[[str], Iterable[ChildChunk]] | None = None):
        """``loader(tenant)`` repopulates a tenant from the durable store on first use."""
        self._loader = loader
        self._chunks: dict[str, dict[str, ChildChunk]] = {}
        self._built: dict[str, _TenantIndex] = {}
        self._terms: dict[str, list[str]] = {}  # tokenisation is the costly part of a rebuild: do it once per chunk
        self._lock = threading.RLock()

    def _tenant_chunks(self, tenant: str) -> dict[str, ChildChunk]:
        if tenant not in self._chunks:
            self._chunks[tenant] = {c.id: c for c in (self._loader(tenant) if self._loader else [])}
        return self._chunks[tenant]

    def add(self, children: Iterable[ChildChunk]) -> None:
        with self._lock:
            for c in children:
                self._tenant_chunks(c.tenant)[c.id] = c
                self._terms.pop(c.id, None)
                self._built.pop(c.tenant, None)

    def delete_document(self, tenant: str, doc_id: str) -> None:
        with self._lock:
            chunks = self._tenant_chunks(tenant)
            for cid in [i for i, c in chunks.items() if c.doc_id == doc_id]:
                del chunks[cid]
                self._terms.pop(cid, None)
            self._built.pop(tenant, None)

    def _index(self, tenant: str) -> _TenantIndex:
        with self._lock:
            idx = self._built.get(tenant)
            if idx is None:
                chunks = list(self._tenant_chunks(tenant).values())
                docs = [self._terms.get(c.id) or self._terms.setdefault(c.id, terms(c.text)) for c in chunks]
                nonempty = any(docs)
                idx = _TenantIndex(chunks, [frozenset(d) for d in docs], BM25Plus(docs) if nonempty else None)
                self._built[tenant] = idx
            return idx

    def search(self, query: str, tenant: str, groups: frozenset[str], k: int) -> list[tuple[ChildChunk, float]]:
        idx = self._index(tenant)
        q = terms(query)
        if idx.bm25 is None or not q:
            return []
        scores = idx.bm25.get_scores(q)
        qset = set(q)
        hits = [(idx.chunks[i], float(scores[i])) for i in range(len(idx.chunks))
                if idx.termsets[i] & qset and visible(idx.chunks[i].acl, groups)]
        return sorted(hits, key=lambda x: -x[1])[:k]
