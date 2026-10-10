"""Hybrid retrieval: dense (vector store) + sparse (BM25) -> RRF -> cross-encoder rerank -> parent chunks."""
from __future__ import annotations

from typing import Sequence

from ..core.config import Settings
from ..core.models import ChildChunk, Context, ParentChunk
from .bm25 import BM25Index
from .reranker import Reranker
from .vector_store import VectorStore


def reciprocal_rank_fusion(rankings: Sequence[Sequence[ChildChunk]], k: int = 60) -> list[tuple[ChildChunk, float]]:
    """score(d) = sum over rankings of 1 / (k + rank(d)), rank starting at 1."""
    score: dict[str, float] = {}
    by_id: dict[str, ChildChunk] = {}
    for ranking in rankings:
        for pos, c in enumerate(ranking, start=1):
            by_id[c.id] = c
            score[c.id] = score.get(c.id, 0.0) + 1.0 / (k + pos)
    return [(by_id[i], score[i]) for i in sorted(score, key=lambda i: (-score[i], i))]


class HybridIndex:
    """Keeps the vector store and the BM25 index in sync."""

    def __init__(self, store: VectorStore, bm25: BM25Index | None = None):
        self.store = store
        self.bm25 = bm25 or BM25Index(loader=lambda tenant: store.iter_children(tenant))

    def add(self, parents: Sequence[ParentChunk], children: Sequence[ChildChunk], vectors) -> None:
        self.store.add(parents, children, vectors)
        self.bm25.add(children)

    def delete_document(self, tenant: str, doc_id: str) -> None:
        self.store.delete_document(tenant, doc_id)
        self.bm25.delete_document(tenant, doc_id)

    def health(self) -> bool:
        return self.store.health()


class HybridRetriever:
    def __init__(self, index: HybridIndex, embedder, reranker: Reranker, settings: Settings):
        self.index, self.embedder, self.reranker, self.cfg = index, embedder, reranker, settings

    def search_children(self, query: str, tenant: str, groups: frozenset[str] = frozenset()) -> list[tuple[ChildChunk, float]]:
        qvec = self.embedder.embed([query])[0]
        dense = [c for c, _ in self.index.store.dense(qvec, tenant, groups, self.cfg.dense_k)]
        sparse = [c for c, _ in self.index.bm25.search(query, tenant, groups, self.cfg.sparse_k)]
        return reciprocal_rank_fusion([dense, sparse], self.cfg.rrf_k)

    def retrieve(self, query: str, tenant: str, groups: frozenset[str] = frozenset(),
                 top_k: int | None = None) -> list[Context]:
        """Top-K parent chunks, ordered by the best reranker score of any of their children."""
        fused = self.search_children(query, tenant, groups)[: self.cfg.rerank_candidates]
        if not fused:
            return []
        scores = self.reranker.score(query, [c.text for c, _ in fused])
        best: dict[str, float] = {}
        for (child, _), s in zip(fused, scores):
            best[child.parent_id] = max(s, best.get(child.parent_id, float("-inf")))
        floor = getattr(self.reranker, "min_score", None)
        contexts: list[Context] = []
        for pid in sorted(best, key=lambda p: -best[p]):
            if floor is not None and best[pid] <= floor:
                continue
            parent = self.index.store.parent(pid, tenant)
            if parent is not None:
                contexts.append(Context(len(contexts) + 1, parent, best[pid]))
            if len(contexts) >= (top_k or self.cfg.top_k):
                break
        return contexts
