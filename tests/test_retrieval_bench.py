"""Retrieval benchmark: metric definitions, dataset integrity and determinism, and the recorded regression floor."""
import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from contextrag.eval import retrieval_bench as B
from contextrag.eval.bench_corpus import DATA_PATH, build_bench

ROOT = Path(__file__).resolve().parents[1]


# ---- metrics, checked by hand ----------------------------------------------------------------------

RANKED = ["a", "b", "c", "d", "e"]


def test_recall_and_precision_at_k():
    rel = {"b", "d", "z"}  # z is never retrieved
    assert B.recall_at_k(RANKED, rel, 2) == pytest.approx(1 / 3)
    assert B.recall_at_k(RANKED, rel, 5) == pytest.approx(2 / 3)
    assert B.precision_at_k(RANKED, rel, 4) == pytest.approx(2 / 4)
    assert B.precision_at_k(RANKED, rel, 1) == 0.0


def test_reciprocal_rank_is_the_inverse_rank_of_the_first_relevant_document():
    assert B.reciprocal_rank(RANKED, {"c", "e"}) == pytest.approx(1 / 3)
    assert B.reciprocal_rank(RANKED, {"a"}) == 1.0
    assert B.reciprocal_rank(RANKED, {"zz"}) == 0.0


def test_ndcg_matches_the_textbook_formula_with_binary_gains():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)  # relevant at ranks 2 and 4
    ideal = 1 / math.log2(2) + 1 / math.log2(3)  # both at the top
    assert B.ndcg_at_k(RANKED, {"b", "d"}, 5) == pytest.approx(dcg / ideal)
    assert B.ndcg_at_k(["b", "d"], {"b", "d"}, 5) == pytest.approx(1.0)
    assert B.ndcg_at_k(RANKED, {"zz"}, 5) == 0.0
    # more relevant documents than k: the ideal is capped at k, so a perfect top-k scores 1
    assert B.ndcg_at_k(["a", "b"], {"a", "b", "c", "d"}, 2) == pytest.approx(1.0)


def test_percentiles_use_nearest_rank_and_p99_needs_enough_samples():
    values = list(range(1, 101))  # 1..100
    assert B.percentile(values, 50) == 50 and B.percentile(values, 95) == 95 and B.percentile(values, 99) == 99
    small = B.latency_summary([1.0, 2.0, 3.0])
    assert small["p99_ms"] is None and small["n"] == 3  # with 3 samples p99 would just be the maximum
    assert B.latency_summary([float(i) for i in range(1, 101)])["p99_ms"] == 99.0


# ---- dataset ---------------------------------------------------------------------------------------------

def test_the_checked_in_dataset_is_exactly_what_the_generator_produces():
    expected = json.dumps(build_bench(), indent=1, ensure_ascii=False) + "\n"
    assert DATA_PATH.read_text(encoding="utf-8") == expected
    assert build_bench() == build_bench()  # deterministic


def test_the_dataset_is_large_enough_and_internally_consistent():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    ids = [d["doc_id"] for d in data["documents"]]
    assert len(ids) == len(set(ids)) == 40
    queries = data["queries"]
    assert len(queries) >= 50 and len({q["id"] for q in queries}) == len(queries)
    kinds = {q["kind"] for q in queries}
    assert kinds == {"direct", "paraphrase", "multi", "unanswerable"}
    for q in queries:
        assert set(q["relevant"]) <= set(ids)
        assert (q["kind"] == "unanswerable") == (not q["relevant"])
    assert data["synthetic"] is True


def test_a_different_seed_changes_the_dataset():
    assert build_bench(seed=12) != build_bench(seed=11)


# ---- benchmark run and regression floor ------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    return B.run_benchmark()


def test_the_report_names_its_inputs_and_what_it_did_not_measure(report):
    assert report["dataset"]["sha256"] == json.loads((B.BASELINE_PATH).read_text())["dataset_sha256"]
    assert report["components"]["embedder"] == "HashingEmbedder"
    assert "not production" in report["caveat"] or "not production retrieval quality" in report["caveat"]
    assert any("token" in item for item in report["not_measured"])
    assert set(report["retrieval"]) == set(B.STRATEGIES)
    assert report["indexing"]["failures"] == 0


def test_no_metric_has_fallen_below_the_recorded_floor(report):
    baseline = json.loads(B.BASELINE_PATH.read_text(encoding="utf-8"))
    assert B.regressions(report, baseline) == []


def test_results_are_reproducible_apart_from_latency(report):
    again = B.run_benchmark()
    for name in B.STRATEGIES:
        assert again["retrieval"][name]["metrics"] == report["retrieval"][name]["metrics"]
    assert again["grounding"]["answerable"] == report["grounding"]["answerable"]
    assert again["grounding"]["unanswerable"] == report["grounding"]["unanswerable"]


def test_the_system_never_returns_a_verified_but_wrong_answer_on_this_corpus(report):
    g = report["grounding"]
    assert g["answerable"]["verified_wrong_rate"] == 0.0
    assert g["unanswerable"]["answered_as_supported_rate"] == 0.0


def test_reranking_improves_on_plain_fusion_but_not_on_bm25_alone_here(report):
    """Recorded as measured, not as a hope: on this lexical corpus fusion with a hashing embedder is worse than BM25
    alone, and the reranker closes most of that gap without exceeding it."""
    m = {name: report["retrieval"][name]["metrics"]["all"] for name in B.STRATEGIES}
    assert m["hybrid_rerank"]["ndcg@5"] > m["hybrid"]["ndcg@5"]
    assert m["hybrid_rerank"]["recall@5"] > m["hybrid"]["recall@5"]
    assert m["hybrid_rerank"]["ndcg@5"] <= m["bm25"]["ndcg@5"] + 1e-9


def test_the_gate_detects_a_regression_in_either_direction(report):
    baseline = json.loads(B.BASELINE_PATH.read_text(encoding="utf-8"))
    worse = copy.deepcopy(report)
    worse["retrieval"]["hybrid_rerank"]["metrics"]["all"]["recall@5"] = 0.5
    worse["grounding"]["answerable"]["verified_wrong_rate"] = 0.2
    problems = B.regressions(worse, baseline)
    assert any(p.startswith("hybrid_rerank.recall@5") for p in problems)
    assert any("verified_wrong_rate" in p for p in problems)


def test_the_cli_check_passes_and_a_missing_baseline_is_a_clean_error(tmp_path):
    ok = subprocess.run([sys.executable, "-m", "contextrag", "bench", "--check-baseline"], cwd=ROOT,
                        capture_output=True, text=True, timeout=300)
    assert ok.returncode == 0, ok.stderr
    assert "hybrid_rerank" in ok.stdout
    bad = subprocess.run([sys.executable, "-m", "contextrag", "bench", "--repeats", "0"], cwd=ROOT,
                         capture_output=True, text=True, timeout=300)
    assert bad.returncode == 2 and "--repeats" in bad.stderr
