"""Injection quarantine, measured: what it catches, what it misses (recorded), and what it wrongly withholds."""
from contextrag.eval.injection_bench import run_injection_bench


def test_the_quarantine_catches_the_pattern_families_it_was_written_for():
    r = run_injection_bench()
    by = r["attacks"]["by_kind"]
    for kind in ("override_en", "override_es", "role_marker", "role_tag", "reveal_secret", "conceal", "exfiltrate"):
        assert by[kind]["flagged"] == by[kind]["attacks"], kind


def test_paraphrased_attacks_are_missed_and_that_gap_is_pinned_not_hidden():
    r = run_injection_bench()
    assert r["attacks"]["by_kind"]["paraphrase"] == {"attacks": 4, "flagged": 0}
    assert r["attacks"]["detection_rate"] == 0.7647  # 13 of 17 synthetic attacks


def test_no_legitimate_document_in_the_set_is_withheld():
    r = run_injection_bench()
    assert r["legitimate"]["false_positive_rate"] == 0.0
    assert r["legitimate"]["wrongly_flagged"] == []


def test_the_report_says_it_is_not_a_classifier():
    assert "not a classifier" in run_injection_bench()["caveat"]
