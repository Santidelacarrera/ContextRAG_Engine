from __future__ import annotations

from typing import Protocol, Sequence

from ..core.text import content_tokens, stem


class Reranker(Protocol):
    """Optional attribute ``min_score``: parents scoring <= it are discarded (cross-encoder
    logits are unbounded, so none is applied unless the reranker declares one)."""

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class LexicalReranker:
    """Offline stand-in: fraction of query terms covered by the passage."""

    min_score = 0.0  # passages with no query-term overlap are dropped (score must be > min_score)

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
