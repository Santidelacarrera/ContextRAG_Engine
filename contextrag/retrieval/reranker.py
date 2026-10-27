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
    def __init__(self, model: str = "BAAI/bge-reranker-large", min_score: float | None = None):
        """``min_score`` (logit) enables relevance gating of retrieved parents and of answer evidence."""
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(model)
        self.model_name = model  # part of the answer-cache key
        self.min_score = min_score

    def score(self, query, passages):
        return [float(s) for s in self._m.predict([(query, p) for p in passages])]


class RerankerError(RuntimeError):
    """A reranker failed or answered with something that cannot be mapped back to the candidates."""


class CohereReranker:
    def __init__(self, api_key: str | None = None, model: str = "rerank-v3.5", min_score: float | None = None):
        import cohere

        self._c, self._model, self.min_score = cohere.ClientV2(api_key), model, min_score
        self.model_name = model  # part of the answer-cache key

    def score(self, query, passages):
        if not passages:
            return []
        r = self._c.rerank(model=self._model, query=query, documents=list(passages), top_n=len(passages))
        # Every candidate must receive exactly one score. A missing index must not default to 0.0: that would
        # silently rank an unscored passage as irrelevant, or worse, as relevant if the default were high.
        out: list[float | None] = [None] * len(passages)
        for item in r.results:
            if not (0 <= item.index < len(passages)) or out[item.index] is not None:
                raise RerankerError(f"cohere returned an invalid or duplicate index {item.index}")
            out[item.index] = float(item.relevance_score)
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            raise RerankerError(f"cohere returned no score for {len(missing)} of {len(passages)} candidates")
        return out  # type: ignore[return-value]  # every slot is filled, checked above


class BoundedReranker:
    """Wraps any reranker with a hard timeout and a bounded worker pool, and validates the score count.

    Policy: a timeout or a failure raises ``RerankerError``. Nothing is guessed, so the request fails instead of
    being answered with a ranking that was never computed. A timed-out call keeps running in its worker thread
    (Python cannot kill it), which is why the pool is bounded and the wrapper is the only entry point."""

    def __init__(self, inner: Reranker, timeout_s: float, *, max_workers: int = 2):
        from concurrent.futures import ThreadPoolExecutor

        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        self._inner = inner
        self._timeout = timeout_s
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="rerank")

    @property
    def min_score(self):
        return getattr(self._inner, "min_score", None)

    @property
    def model_name(self):
        """Identity of the wrapped model, so the answer cache distinguishes checkpoints behind the same wrapper."""
        return f"{type(self._inner).__name__}:{getattr(self._inner, 'model_name', None)}"

    def score(self, query, passages):
        from concurrent.futures import TimeoutError as FutureTimeout

        if not passages:
            return []
        future = self._pool.submit(self._inner.score, query, list(passages))
        try:
            scores = future.result(timeout=self._timeout)
        except FutureTimeout:
            future.cancel()
            raise RerankerError(f"reranker timed out after {self._timeout:g}s") from None
        except RerankerError:
            raise
        except Exception as e:
            raise RerankerError(f"reranker failed: {type(e).__name__}") from e
        if len(scores) != len(passages):
            raise RerankerError(f"reranker returned {len(scores)} scores for {len(passages)} candidates")
        return list(scores)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
