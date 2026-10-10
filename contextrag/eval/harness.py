"""Evaluation harness: measures how often the system is wrong *and confident* -- the failure the
product exists to prevent.

* ``eval_verifier``: labelled (premise, claim) pairs. ``false_support_rate`` = share of *unsupported*
  claims that the NLI model scores >= threshold (must be 0 for a safe configuration);
  ``recall`` = share of supported claims accepted (a usability measure).
* ``eval_rag``: end-to-end questions over a small corpus. ``unsafe_rate`` = verified answers to
  unanswerable questions or answers missing the expected fact.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Callable

from ..core.models import Status
from ..pipeline import ContextRAG
from ..verification.nli_verifier import NLIModel, verify_attribution


def load_json(name: str, path: str | None = None):
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return json.loads(resources.files("contextrag.eval").joinpath("data", name).read_text(encoding="utf-8"))


@dataclass
class VerifierReport:
    threshold: float
    total: int
    false_support_rate: float
    recall: float
    false_supports: list[str] = field(default_factory=list)
    missed_supports: list[str] = field(default_factory=list)
    by_category: dict[str, float] = field(default_factory=dict)  # accuracy per category

    def to_dict(self):
        return asdict(self)


def eval_verifier(nli: NLIModel, threshold: float = 0.95, cases: list[dict] | None = None) -> VerifierReport:
    cases = cases if cases is not None else load_json("verifier_cases.json")
    false_sup, missed, cat = [], [], {}
    pos = neg = 0
    for c in cases:
        accepted = verify_attribution(c["claim"], [c["premise"]], nli) >= threshold
        good = accepted == (c["label"] == "supported")
        cat.setdefault(c["category"], []).append(good)
        if c["label"] == "supported":
            pos += 1
            if not accepted:
                missed.append(c["id"])
        else:
            neg += 1
            if accepted:
                false_sup.append(c["id"])
    return VerifierReport(
        threshold, len(cases), len(false_sup) / neg if neg else 0.0, 1 - len(missed) / pos if pos else 1.0,
        false_sup, missed, {k: sum(v) / len(v) for k, v in cat.items()})


@dataclass
class RagReport:
    total: int
    answer_rate: float  # answerable questions that got a verified answer
    correct_rate: float  # ... that contain the expected fact
    abstention_rate: float  # unanswerable questions that got the fallback
    unsafe_rate: float  # verified-but-wrong or verified-on-unanswerable, over all cases
    retrieval_hit_rate: float  # expected document among the retrieved sources
    mrr: float  # mean reciprocal rank of the expected document
    unsafe: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def eval_rag(make_rag: Callable[[], ContextRAG], data: dict | None = None, tenant: str = "eval") -> RagReport:
    data = data or load_json("rag_cases.json")
    rag = make_rag()
    for d in data["documents"]:
        rag.ingestor.ingest_text(d["text"], tenant=tenant, doc_id=d["doc_id"], source=d["doc_id"])
    ans = cor = abst = hits = 0
    rr: list[float] = []
    n_ans = sum(c["answerable"] for c in data["cases"])
    n_un = len(data["cases"]) - n_ans
    unsafe, failures = [], []
    for c in data["cases"]:
        a = rag.ask(c["question"], tenant=tenant)
        verified = a.status is Status.VERIFIED
        if c["answerable"]:
            docs = [s.parent.doc_id for s in a.sources]
            if c["doc"] in docs:
                hits += 1
                rr.append(1 / (docs.index(c["doc"]) + 1))
            else:
                rr.append(0.0)
            if verified:
                ans += 1
                if c["expect"].lower() in a.text.lower():
                    cor += 1
                else:
                    unsafe.append(c["id"])
            else:
                failures.append(c["id"])
        else:
            if verified:
                unsafe.append(c["id"])
            else:
                abst += 1
    return RagReport(
        len(data["cases"]), ans / n_ans if n_ans else 1.0, cor / n_ans if n_ans else 1.0,
        abst / n_un if n_un else 1.0, len(unsafe) / len(data["cases"]), hits / n_ans if n_ans else 1.0,
        sum(rr) / len(rr) if rr else 0.0, unsafe, failures)
