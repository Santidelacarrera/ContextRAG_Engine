"""Reranking must never invent a score: incomplete responses fail, timeouts fail, and remote providers are bounded."""
import time
from types import SimpleNamespace

import pytest

from contextrag.retrieval.reranker import BoundedReranker, CohereReranker, RerankerError


class FakeCohere:
    def __init__(self, results):
        self.results = results

    def rerank(self, **kwargs):
        return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in self.results])


def cohere_with(results):
    r = CohereReranker.__new__(CohereReranker)  # bypass the SDK import: the call surface is what is under test
    r._c, r._model, r.min_score = FakeCohere(results), "rerank-test", None
    return r


def test_cohere_maps_scores_back_to_their_candidates_by_index():
    assert cohere_with([(1, 0.2), (0, 0.9)]).score("q", ["a", "b"]) == [0.9, 0.2]


def test_a_missing_candidate_score_is_an_error_not_a_zero():
    with pytest.raises(RerankerError, match="no score for 1 of 2"):
        cohere_with([(0, 0.9)]).score("q", ["a", "b"])


@pytest.mark.parametrize("bad", [[(5, 0.1)], [(0, 0.1), (0, 0.2)]])
def test_out_of_range_or_duplicate_indexes_are_rejected(bad):
    with pytest.raises(RerankerError):
        cohere_with(bad).score("q", ["a", "b"])


def test_empty_candidates_return_empty_without_calling_the_provider():
    assert cohere_with([]).score("q", []) == []


class Slow:
    min_score = 0.5

    def __init__(self, delay, scores=None):
        self.delay, self.scores = delay, scores

    def score(self, query, passages):
        time.sleep(self.delay)
        return self.scores if self.scores is not None else [1.0] * len(passages)


def test_a_slow_reranker_times_out_and_the_request_fails_instead_of_guessing():
    b = BoundedReranker(Slow(delay=2.0), timeout_s=0.1)
    try:
        with pytest.raises(RerankerError, match="timed out"):
            b.score("q", ["a"])
    finally:
        b.close()


def test_wrong_score_count_from_the_inner_reranker_is_rejected():
    b = BoundedReranker(Slow(0, scores=[1.0]), timeout_s=5)
    try:
        with pytest.raises(RerankerError, match="returned 1 scores for 2"):
            b.score("q", ["a", "b"])
    finally:
        b.close()


def test_inner_exceptions_are_wrapped_with_their_type_only():
    class Broken:
        def score(self, query, passages):
            raise ConnectionError("https://secret-host/key=abc")

    b = BoundedReranker(Broken(), timeout_s=5)
    try:
        with pytest.raises(RerankerError) as info:
            b.score("q", ["a"])
    finally:
        b.close()
    assert "ConnectionError" in str(info.value) and "secret" not in str(info.value)


def test_the_wrapper_keeps_the_relevance_floor_and_scores_on_success():
    b = BoundedReranker(Slow(0, scores=[0.7, 0.1]), timeout_s=5)
    try:
        assert b.min_score == 0.5
        assert b.score("q", ["a", "b"]) == [0.7, 0.1]
    finally:
        b.close()
