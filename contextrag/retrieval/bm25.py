"""Local sparse retrieval with ``rank_bm25`` (BM25Plus), kept per tenant.

BM25Plus is used instead of BM25Okapi because Okapi's IDF turns negative when a term
occurs in more than half of the documents, which breaks small corpora. The index is
immutable in rank_bm25, so it is rebuilt lazily after any change.
"""
from __future__ import annotations

import math
import threading
from collections import Counter
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

    def delete_tenant(self, tenant: str) -> None:
        with self._lock:
            for cid in list(self._chunks.get(tenant, {})):
                self._terms.pop(cid, None)
            self._chunks[tenant] = {}  # an explicit empty set: do not lazily reload from the (already emptied) store
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
        return sorted(hits, key=lambda x: (-x[1], x[0].id))[:k]


class IncrementalBM25Index:
    """BM25Plus (same formula and defaults as ``rank_bm25.BM25Plus``: k1=1.5, b=0.75, delta=1) over an inverted
    index that is updated in place: adding or deleting a document costs O(its terms), and a query only touches the
    postings of its own terms. Unlike ``BM25Index`` (rank_bm25 is immutable and is rebuilt after every change) this
    stays fast under frequent ingestion on large tenants. Scores equal rank_bm25's (see tests)."""

    def __init__(self, loader: Callable[[str], Iterable[ChildChunk]] | None = None,
                 k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self._loader, self.k1, self.b, self.delta = loader, k1, b, delta
        self._t: dict[str, _Inc] = {}
        self._lock = threading.RLock()

    def _tenant(self, tenant: str) -> "_Inc":
        if tenant not in self._t:
            inc = self._t[tenant] = _Inc()
            for c in (self._loader(tenant) if self._loader else []):
                inc.add(c)
        return self._t[tenant]

    def add(self, children: Iterable[ChildChunk]) -> None:
        with self._lock:
            for c in children:
                self._tenant(c.tenant).add(c)

    def delete_document(self, tenant: str, doc_id: str) -> None:
        with self._lock:
            inc = self._tenant(tenant)
            for cid in [i for i, c in inc.chunks.items() if c.doc_id == doc_id]:
                inc.remove(cid)

    def delete_tenant(self, tenant: str) -> None:
        with self._lock:
            self._t[tenant] = _Inc()

    def search(self, query: str, tenant: str, groups: frozenset[str], k: int) -> list[tuple[ChildChunk, float]]:
        q = terms(query)
        with self._lock:
            inc = self._tenant(tenant)
            n = len(inc.chunks)
            if not n or not q:
                return []
            avgdl = inc.total_len / n or 1.0
            scores: dict[str, float] = {}
            offset = 0.0  # delta * idf, added for every query term, as rank_bm25 does (also for absent terms)
            for term, qf in Counter(q).items():
                post = inc.postings.get(term)
                if not post:
                    continue
                idf = math.log((n + 1) / len(post))
                offset += qf * self.delta * idf
                for cid, tf in post.items():
                    norm = self.k1 * (1 - self.b + self.b * inc.dl[cid] / avgdl) + tf
                    scores[cid] = scores.get(cid, 0.0) + qf * idf * (tf * (self.k1 + 1) / norm)
            hits = [(inc.chunks[cid], s + offset) for cid, s in scores.items() if visible(inc.chunks[cid].acl, groups)]
        return sorted(hits, key=lambda x: (-x[1], x[0].id))[:k]


class _Inc:
    def __init__(self):
        self.chunks: dict[str, ChildChunk] = {}
        self.tf: dict[str, Counter] = {}
        self.dl: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = {}
        self.total_len = 0

    def add(self, c: ChildChunk) -> None:
        if c.id in self.chunks:
            self.remove(c.id)
        tf = Counter(terms(c.text))
        self.chunks[c.id], self.tf[c.id] = c, tf
        self.dl[c.id] = sum(tf.values())
        self.total_len += self.dl[c.id]
        for t, f in tf.items():
            self.postings.setdefault(t, {})[c.id] = f

    def remove(self, cid: str) -> None:
        self.total_len -= self.dl.pop(cid)
        for t in self.tf.pop(cid):
            post = self.postings[t]
            del post[cid]
            if not post:
                del self.postings[t]
        del self.chunks[cid]
