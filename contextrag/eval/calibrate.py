"""Calibration of chunk sizes, the reranker relevance floor and the NLI verification threshold.

What is calibrated, and on what:

* **Chunk grid** ``parent {1024, 2000} x child {256, 400}``: each combination is ingested and scored with
  :func:`evaluate_context` (lexical context recall and average precision, the same metrics as the regression
  baseline). Selection policy: recall first (no candidate may lose recall against the best one by more than
  ``recall_tolerance``, default 0), then highest average precision, then the smaller configuration. The policy is
  written into the report. At the defaults, a single lost case out of 18 is a 0.056 recall difference, which is
  why the tolerance is zero and why any change must be read against the sample size.
* **``RERANK_MIN_SCORE``**: precision-recall over the reranker's scores on the candidates returned for each answerable
  case. A candidate is relevant when its parent comes from the case's labelled document (``doc``). Selection policy:
  among thresholds whose precision reaches ``target_precision``, the one with the highest recall, ties to the lower
  threshold. If no threshold reaches the target, the recommendation is ``null``.
* **NLI threshold**: :func:`contextrag.eval.harness.sweep_thresholds` over statement-level labels (premise, claim,
  supported or not). Selection: the lowest threshold with zero false supports.

Trust rules, enforced in code and stated in the report:

* A section whose labelled set is below ``min_per_class`` in either class is marked ``insufficient_data`` and its
  recommendation is still shown but flagged. Small samples do not justify a threshold.
* The report records ``data_kind``: ``synthetic`` (the deterministic long corpus), ``fixture`` (bundled), or
  ``supplied`` (a file the operator provided). Only ``supplied`` real documents can validate a production choice.
* Results are deterministic for a given input and seed. The report records the input file's SHA-256 and the seed.
* Nothing is written to production configuration. The only file written is the explicit ``--proposal`` path, and it
  holds candidate values for review, never ``.env`` or ``Settings`` defaults.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.config import Settings
from .context_metrics import evaluate_context

PARENT_GRID = (1024, 2000)
CHILD_GRID = (256, 400)
MIN_PER_CLASS = 50  # below this, a threshold is not supported by the data (rule of thumb, stated in the report)
DATA_KINDS = ("synthetic", "fixture", "supplied")


class CalibrationError(RuntimeError):
    """A fatal problem with the inputs or the computation. The CLI exits nonzero and writes no proposal."""


@dataclass(frozen=True)
class ChunkResult:
    parent_tokens: int
    child_tokens: int
    cases: int
    mean_recall: float
    mean_average_precision: float


@dataclass
class Section:
    name: str
    status: str  # computed | insufficient_data | not_run
    data: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _answerable(data: dict) -> list[dict]:
    cases = [c for c in data["cases"] if c.get("answerable")]
    if not cases:
        raise CalibrationError("no answerable cases: nothing to calibrate against")
    return cases


def chunk_grid(data: dict, *, tenant: str = "calibrate", base: Settings | None = None,
               recall_tolerance: float = 0.0) -> tuple[list[ChunkResult], dict]:
    """Score every (parent, child) combination on the same documents and cases."""
    from ..factory import offline

    base = base or Settings()
    rows: list[ChunkResult] = []
    for parent, child in itertools.product(PARENT_GRID, CHILD_GRID):
        if child > parent:
            continue  # the chunker refuses child > parent; no such combination exists in the grid
        cfg = replace(base, parent_tokens=parent, child_tokens=child,
                      child_overlap_tokens=min(base.child_overlap_tokens, child - 1))
        report = evaluate_context(lambda cfg=cfg: offline(cfg), data, tenant=tenant)
        rows.append(ChunkResult(parent, child, report.total, report.mean_recall, report.mean_average_precision))
    # Selection policy. Recall is a ceiling: a fact that is not in the evidence cannot be answered, whatever the
    # ranking. So the candidates are first restricted to those within ``recall_tolerance`` of the best recall, and
    # only then ranked by average precision, with the smaller configuration as the final tie-break.
    best_recall = max(r.mean_recall for r in rows)
    eligible = [r for r in rows if r.mean_recall >= best_recall - recall_tolerance]
    best = max(eligible, key=lambda r: (r.mean_average_precision, -r.parent_tokens, -r.child_tokens))
    return rows, {"policy": (f"recall >= best - {recall_tolerance:g}, then max mean_average_precision, then smaller "
                             "parent, then smaller child"),
                  "recommended": {"parent_tokens": best.parent_tokens, "child_tokens": best.child_tokens}}


def _precision_recall(scores: Sequence[tuple[float, bool]], threshold: float) -> tuple[int, int, int]:
    tp = sum(1 for s, rel in scores if s >= threshold and rel)
    fp = sum(1 for s, rel in scores if s >= threshold and not rel)
    fn = sum(1 for s, rel in scores if s < threshold and rel)
    return tp, fp, fn


def rerank_threshold(data: dict, *, tenant: str = "calibrate", target_precision: float = 0.9,
                     min_per_class: int = MIN_PER_CLASS) -> Section:
    """Precision-recall of the reranker floor on the candidates each answerable case receives."""
    from ..factory import offline

    if not 0.0 < target_precision <= 1.0:
        raise CalibrationError("target_precision must be in (0, 1]")
    rag = offline()
    for d in data["documents"]:
        rag.ingestor.ingest_text(d["text"], tenant=tenant, doc_id=d["doc_id"], source=d["doc_id"])
    labelled: list[tuple[float, bool]] = []
    for case in _answerable(data):
        if "doc" not in case:
            raise CalibrationError(f"case {case.get('id')!r} has no labelled document ('doc')")
        detailed = rag.retriever.retrieve_detailed(case["question"], tenant)
        if not detailed:
            continue
        passages = [ctx.parent.text for ctx, _ in detailed]
        scores = rag.reranker.score(case["question"], passages)
        for (ctx, _), s in zip(detailed, scores, strict=True):
            labelled.append((float(s), ctx.parent.doc_id == case["doc"]))
    pos = sum(rel for _, rel in labelled)
    neg = len(labelled) - pos
    if not labelled:
        return Section("rerank_min_score", "not_run", {"reason": "no candidates were retrieved"})
    thresholds = sorted({s for s, _ in labelled})
    if len(thresholds) > 200:  # bounded grid: quantiles of the observed scores
        thresholds = sorted({thresholds[int(i * (len(thresholds) - 1) / 199)] for i in range(200)})
    rows = []
    for t in thresholds:
        tp, fp, fn = _precision_recall(labelled, t)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        rows.append({"threshold": round(t, 6), "precision": round(precision, 4), "recall": round(recall, 4),
                     "tp": tp, "fp": fp, "fn": fn})
    eligible = [r for r in rows if r["precision"] >= target_precision and r["tp"] > 0]
    recommended = (min(eligible, key=lambda r: (-r["recall"], r["threshold"]))["threshold"] if eligible else None)
    insufficient = pos < min_per_class or neg < min_per_class
    return Section("rerank_min_score", "insufficient_data" if insufficient else "computed", {
        "policy": f"lowest-threshold-tie highest recall subject to precision >= {target_precision}",
        "target_precision": target_precision, "candidates": len(labelled), "relevant": pos, "irrelevant": neg,
        "min_per_class": min_per_class, "recommended": recommended, "rows": rows})


def nli_threshold(verifier_cases: list[dict], *, min_per_class: int = MIN_PER_CLASS) -> Section:
    """Statement-level NLI sweep. Uses the lexical entailment model; a DeBERTa threshold needs its own labels."""
    from .harness import sweep_thresholds
    from ..verification.nli_verifier import LexicalEntailment

    if not verifier_cases:
        return Section("nli_threshold", "not_run", {"reason": "no statement-level labels supplied"})
    pos = sum(c["label"] == "supported" for c in verifier_cases)
    neg = len(verifier_cases) - pos
    sweep = sweep_thresholds(LexicalEntailment(), verifier_cases)
    insufficient = pos < min_per_class or neg < min_per_class
    return Section("nli_threshold", "insufficient_data" if insufficient else "computed", {
        "policy": "lowest threshold with zero false supports on the labelled set",
        "labelled": {"supported": pos, "unsupported": neg}, "min_per_class": min_per_class,
        "nli_model": "lexical (not the production DeBERTa model)", **sweep})


def calibrate(data: dict, *, verifier_cases: list[dict] | None, data_kind: str, data_path: str | None,
              seed: int | None, target_precision: float = 0.9, tenant: str = "calibrate") -> dict[str, Any]:
    if data_kind not in DATA_KINDS:
        raise CalibrationError(f"data_kind must be one of {DATA_KINDS}")
    rows, chunk_section = chunk_grid(data, tenant=tenant)
    rerank = rerank_threshold(data, tenant=tenant, target_precision=target_precision)
    nli = nli_threshold(verifier_cases or [])
    import contextrag

    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"kind": data_kind, "path": data_path,
                 "sha256": sha256_file(data_path) if data_path else None,
                 "documents": len(data["documents"]), "answerable_cases": len(_answerable(data)),
                 "seed": seed, "verifier_labels": len(verifier_cases or [])},
        "versions": {"contextrag": getattr(contextrag, "__version__", "unknown"), "python": platform.python_version()},
        "chunk_grid": {"parent_tokens": list(PARENT_GRID), "child_tokens": list(CHILD_GRID),
                       "rows": [asdict(r) for r in rows],
                       "selection": chunk_section["policy"], "recommended": chunk_section["recommended"],
                       "status": "computed", "evidence_level": data_kind},
        "rerank_min_score": asdict(rerank) | {"data_kind": data_kind},
        "nli_threshold": asdict(nli),
        "claims": ("Verified only on the data named above. Synthetic or fixture results are regression evidence, "
                   "not evidence about production documents. Production values need supplied, labelled data."),
    }


def write_proposal(report: dict, path: str | Path) -> None:
    """Candidate values for review. Never read back by the application."""
    proposal = {
        "NOTE": "candidates for review; not applied to any configuration",
        "CONTEXTRAG_PARENT_TOKENS": report["chunk_grid"]["recommended"]["parent_tokens"],
        "CONTEXTRAG_CHILD_TOKENS": report["chunk_grid"]["recommended"]["child_tokens"],
        "RERANK_MIN_SCORE": report["rerank_min_score"]["data"].get("recommended"),
        "statuses": {"chunk_grid": report["chunk_grid"]["status"], "rerank": report["rerank_min_score"]["status"],
                     "nli": report["nli_threshold"]["status"]},
        "evidence_level": report["data"]["kind"],
    }
    Path(path).write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")


__all__ = ["calibrate", "chunk_grid", "rerank_threshold", "nli_threshold", "write_proposal", "CalibrationError"]
