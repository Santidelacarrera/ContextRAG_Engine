"""Chunk stores: hybrid (dense + sparse) retrieval over child chunks, parent lookup."""
from __future__ import annotations

import math
from collections import Counter
from typing import Protocol, Sequence

from .models import ChildChunk, ParentChunk
from .security import visible
from .text import content_tokens, stem


def _terms(text: str) -> list[str]:
    return [stem(t) for t in content_tokens(text)]


class ChunkStore(Protocol):
    def add(
        self, parents: Sequence[ParentChunk], children: Sequence[ChildChunk], vectors: Sequence[Sequence[float]]
    ) -> None: ...

    def dense(self, vec: Sequence[float], tenant: str, groups: frozenset[str], k: int) -> list[tuple[ChildChunk, float]]: ...

    def sparse(self, query: str, tenant: str, groups: frozenset[str], k: int) -> list[tuple[ChildChunk, float]]: ...

    def parent(self, parent_id: str, tenant: str) -> ParentChunk | None: ...

    def delete_document(self, tenant: str, doc_id: str) -> None: ...


class InMemoryStore:
    """Reference implementation: cosine dense search + Okapi BM25."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._parents: dict[str, ParentChunk] = {}
        self._children: dict[str, ChildChunk] = {}
        self._vecs: dict[str, Sequence[float]] = {}
        self._tf: dict[str, Counter] = {}

    def add(self, parents, children, vectors):
        for p in parents:
            self._parents[p.id] = p
        for c, v in zip(children, vectors, strict=True):
            self._children[c.id] = c
            self._vecs[c.id] = v
            self._tf[c.id] = Counter(_terms(c.text))

    def _candidates(self, tenant, groups):
        return [c for c in self._children.values() if c.tenant == tenant and visible(c.acl, groups)]

    def dense(self, vec, tenant, groups, k):
        scored = [(c, sum(a * b for a, b in zip(vec, self._vecs[c.id]))) for c in self._candidates(tenant, groups)]
        return sorted(scored, key=lambda x: -x[1])[:k]

    def sparse(self, query, tenant, groups, k):
        cands = self._candidates(tenant, groups)
        if not cands:
            return []
        n = len(cands)
        avg = sum(sum(self._tf[c.id].values()) for c in cands) / n or 1.0
        df = Counter(t for c in cands for t in self._tf[c.id])
        q = set(_terms(query))
        out = []
        for c in cands:
            tf, dl, s = self._tf[c.id], sum(self._tf[c.id].values()), 0.0
            for t in q & tf.keys():
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / avg))
            if s > 0:
                out.append((c, s))
        return sorted(out, key=lambda x: -x[1])[:k]

    def parent(self, parent_id, tenant):
        p = self._parents.get(parent_id)
        return p if p and p.tenant == tenant else None

    def delete_document(self, tenant, doc_id):
        for cid in [c.id for c in self._children.values() if c.tenant == tenant and c.doc_id == doc_id]:
            for d in (self._children, self._vecs, self._tf):
                d.pop(cid, None)
        for pid in [p.id for p in self._parents.values() if p.tenant == tenant and p.doc_id == doc_id]:
            self._parents.pop(pid, None)
