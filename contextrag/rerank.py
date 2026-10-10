from __future__ import annotations

from typing import Protocol, Sequence

from .models import ChildChunk
from .text import content_tokens, stem


class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class LexicalReranker:
    """Offline stand-in: fraction of query terms covered by the passage."""

    def score(self, query, passages):
        q = {stem(t) for t in content_tokens(query)}
        if not q:
            return [0.0] * len(passages)
        return [len(q & {stem(t) for t in content_tokens(p)}) / len(q) for p in passages]


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-large"):
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(model)

    def score(self, query, passages):
        return [float(s) for s in self._m.predict([(query, p) for p in passages])]


class CohereReranker:
    def __init__(self, api_key: str | None = None, model: str = "rerank-v3.5"):
        import cohere

        self._c, self._model = cohere.ClientV2(api_key), model

    def score(self, query, passages):
        r = self._c.rerank(model=self._model, query=query, documents=list(passages), top_n=len(passages))
        out = [0.0] * len(passages)
        for item in r.results:
            out[item.index] = item.relevance_score
        return out


def rerank(reranker: Reranker, query: str, chunks: Sequence[ChildChunk], top_n: int) -> list[tuple[ChildChunk, float]]:
    if not chunks:
        return []
    scores = reranker.score(query, [c.text for c in chunks])
    return sorted(zip(chunks, scores), key=lambda x: -x[1])[:top_n]
