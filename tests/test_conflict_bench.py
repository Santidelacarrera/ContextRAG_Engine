"""Conflict benchmark: dataset integrity, the measured confusion matrix, and the blind spots that are recorded as such."""
import json

from contextrag.eval.conflict_bench import run_conflict_bench
from contextrag.eval.conflict_corpus import DATA_PATH, build_conflict_bench


def test_the_checked_in_conflict_dataset_is_what_the_generator_produces():
    assert DATA_PATH.read_text(encoding="utf-8") == json.dumps(build_conflict_bench(), indent=1, ensure_ascii=False) + "\n"


def test_every_label_is_represented_so_no_category_is_silently_absent():
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    labels = {c["label"] for c in data["cases"]}
    assert labels == {"contradiction", "version", "compatible", "complementary", "homonym", "insufficient"}


def test_non_conflicts_are_never_reported_as_conflicts():
    """The cost that matters most for a legal reader: a false alarm sends a person to review a non-issue, and a missed
    contradiction is caught by the review flag on answers. Both are measured; this test pins the false-alarm side."""
    r = run_conflict_bench()
    assert r["overall"]["false_positive_rate"] == 0.0
    assert r["false_alarms"] == []


def test_the_confusion_matrix_is_pinned_so_a_change_in_detection_is_visible():
    r = run_conflict_bench()
    assert r["overall"] == {"tp": 8, "fp": 0, "fn": 3, "tn": 11, "precision": 1.0, "recall": 0.7273,
                            "false_positive_rate": 0.0}


def test_the_misses_are_the_known_blind_spots_and_nothing_else():
    misses = sorted((m["label"], m["note"]) for m in run_conflict_bench()["misses"])
    assert misses == [("contradiction", "party"), ("contradiction", "status"), ("version", "old vs new figure")]


def test_the_report_states_its_sample_size_and_what_it_does_not_measure():
    r = run_conflict_bench()
    assert r["dataset"]["pairs"] == 22
    assert "small-sample" in r["caveat"] and "not semantic" in r["caveat"]
