"""Measures conflict detectors on the labelled pairs of ``conflict_corpus``. Confusion counts per label, not one number.

For each pair the two statements are placed in two different documents and given to ``detector.detect`` together with the
question, exactly as the pipeline does. A pair is predicted as a conflict when the detector returns at least one conflict.

Definitions (positive = a conflict, i.e. label in CONFLICT_LABELS):
    precision = TP / (TP + FP)      recall = TP / (TP + FN)      false positive rate = FP / (FP + TN)
Per label, the count of predicted conflicts is reported, so that a category the detector cannot see shows as zero.
With 22 pairs any rate is a small-sample estimate; the report says so.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..core.models import Context, ParentChunk
from ..verification.conflicts import StructuredConflictDetector
from .conflict_corpus import CONFLICT_LABELS, DATA_PATH


def _context(text: str, doc: str, index: int) -> Context:
    return Context(index, ParentChunk(f"p-{doc}", doc, "bench", doc, text, 0, len(text)), 1.0)


def run_conflict_bench(detector: Any = None, dataset: str | Path = DATA_PATH) -> dict[str, Any]:
    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    det = detector or StructuredConflictDetector()
    per_label: dict[str, Counter] = {}
    misses: list[dict[str, str]] = []
    false_alarms: list[dict[str, str]] = []
    for case in data["cases"]:
        contexts = [_context(case["a"], "doc-a.md", 1), _context(case["b"], "doc-b.md", 2)]
        predicted = bool(det.detect(case["question"], contexts))
        actual = case["label"] in CONFLICT_LABELS
        c = per_label.setdefault(case["label"], Counter())
        c["pairs"] += 1
        c["predicted_conflict"] += predicted
        if actual and predicted:
            c["tp"] += 1
        elif actual:
            c["fn"] += 1
            misses.append({"id": case["id"], "label": case["label"], "note": case["note"]})
        elif predicted:
            c["fp"] += 1
            false_alarms.append({"id": case["id"], "label": case["label"], "note": case["note"]})
        else:
            c["tn"] += 1
    tp = sum(c["tp"] for c in per_label.values())
    fp = sum(c["fp"] for c in per_label.values())
    fn = sum(c["fn"] for c in per_label.values())
    tn = sum(c["tn"] for c in per_label.values())

    def ratio(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    return {
        "dataset": {"version": data["version"], "pairs": len(data["cases"]), "synthetic": data["synthetic"]},
        "detector": getattr(det, "name", type(det).__name__),
        "overall": {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": ratio(tp, tp + fp),
                    "recall": ratio(tp, tp + fn), "false_positive_rate": ratio(fp, fp + tn)},
        "per_label": {k: dict(v) for k, v in per_label.items()},
        "misses": misses,
        "false_alarms": false_alarms,
        "caveat": ("Small labelled set written by the author, not by the detector; rates are small-sample estimates. "
                   "Measures the heuristic on dates and figures, not semantic detection."),
    }
