"""Hybrid search -> RRF fusion -> rerank -> parent expansion."""
from __future__ import annotations

from .config import Settings
from .embeddings import Embedder
from .models import ChildChunk, Context
from .rerank import Reranker, rerank
from .store import ChunkStore


def rrf(rankings: list[list[ChildChunk]], k: int) -> list[ChildChunk]:
    score: dict[str, float] = {}
    by_id: dict[str, ChildChunk] = {}
    for ranking in rankings:
        for pos, c in enumerate(ranking):
            by_id[c.id] = c
            score[c.id] = score.get(c.id, 0.0) + 1.0 / (k + pos + 1)
    return [by_id[i] for i in sorted(score, key=lambda i: -score[i])]


class Retriever:
    def __init__(self, store: ChunkStore, embedder: Embedder, reranker: Reranker, settings: Settings):
        self.store, self.embedder, self.reranker, self.cfg = store, embedder, reranker, settings

    def retrieve(self, query: str, tenant: str, groups: frozenset[str] = frozenset()) -> list[Context]:
        qvec = self.embedder.embed([query])[0]
        dense = [c for c, _ in self.store.dense(qvec, tenant, groups, self.cfg.dense_k)]
        sparse = [c for c, _ in self.store.sparse(query, tenant, groups, self.cfg.sparse_k)]
        fused = rrf([dense, sparse], self.cfg.rrf_k)
        ranked = rerank(self.reranker, query, fused, self.cfg.rerank_top_n)

        contexts: list[Context] = []
        seen: set[str] = set()
        for child, score in ranked:
            if score <= 0 or child.parent_id in seen:
                continue
            parent = self.store.parent(child.parent_id, tenant)
            if parent is None:
                continue
            seen.add(child.parent_id)
            contexts.append(Context(len(contexts) + 1, parent))
            if len(contexts) >= self.cfg.max_contexts:
                break
        return contexts
