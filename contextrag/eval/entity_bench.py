"""Entity-handling benchmark: false answers and false abstentions, measured separately.

Each question has an expected outcome (``answer`` / ``abstain`` / ``clarify``). The observed outcome is ``answer`` (a
verified answer), ``clarify`` (a clarification request) or ``abstain`` (any other fallback). Three configurations run on
the same data so that before and after are comparable:

* ``no_gate``: the entity gate is off (the behaviour before the gate existed).
* ``gate_only``: the gate abstains on any named term the evidence lacks (the previous iteration).
* ``alias_resolution``: the current default; documented aliases are resolved, ambiguous ones ask, unknown ones abstain.

Offline stack only (hashing embedder, lexical reranker and NLI, extractive generator): the figures describe the logic of
the pipeline on synthetic data, not quality with real models or real documents.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..core.config import Settings
from ..core.models import Answer, Status
from .entity_corpus import DATA_PATH

TENANT = "entities"
CONFIGS: dict[str, dict[str, Any]] = {
    "no_gate": {"require_question_entities": False, "resolve_entity_aliases": False},
    "gate_only": {"require_question_entities": True, "resolve_entity_aliases": False},
    "alias_resolution": {"require_question_entities": True, "resolve_entity_aliases": True},
}
BASELINE_PATH = DATA_PATH.with_name("entity_bench_baseline.json")


def observed(answer: Answer) -> str:
    if answer.status is Status.VERIFIED:
        return "answer"
    return "clarify" if answer.meta.get("clarification") else "abstain"


def judge(query: dict, answer: Answer) -> dict[str, Any]:
    """Outcome of one question. ``correct_answer`` is only defined for questions that expect an answer."""
    seen = observed(answer)
    out: dict[str, Any] = {"id": query["id"], "kind": query["kind"], "expected": query["expected"], "observed": seen}
    if query["expected"] == "answer" and seen == "answer":
        cited = {c.evidence.doc_id for c in answer.claims if c.evidence}
        out["correct_answer"] = (query["expect"].lower() in answer.text.lower() and bool(cited)
                                 and cited <= {query["relevant_doc"]})
    return out


def run_config(data: dict, flags: dict[str, Any], make_rag: Callable[[Settings], Any]) -> list[dict]:
    rag = make_rag(replace(Settings(), **flags))
    try:
        for d in data["documents"]:
            rag.ingestor.ingest_text(d["text"], tenant=TENANT, doc_id=d["doc_id"], source=d["doc_id"], acl=tuple(d["acl"]))
        return [judge(q, rag.ask(q["question"], tenant=TENANT, groups=frozenset(q["groups"]))) for q in data["queries"]]
    finally:
        rag.close()


def metrics(results: list[dict]) -> dict[str, Any]:
    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    should_answer = [r for r in results if r["expected"] == "answer"]
    should_not = [r for r in results if r["expected"] != "answer"]
    should_abstain = [r for r in results if r["expected"] == "abstain"]
    should_clarify = [r for r in results if r["expected"] == "clarify"]
    answered = [r for r in should_answer if r["observed"] == "answer"]
    return {
        "questions": {"should_answer": len(should_answer), "should_abstain": len(should_abstain),
                      "should_clarify": len(should_clarify)},
        # answered although the evidence cannot support an answer: the dangerous error
        "false_answer_rate": rate(sum(r["observed"] == "answer" for r in should_not), len(should_not)),
        # did not answer although the evidence supports one: the usability error
        "false_abstention_rate": rate(sum(r["observed"] != "answer" for r in should_answer), len(should_answer)),
        # answered, but with a wrong value or a citation outside the expected document
        "wrong_answer_rate": rate(sum(not r["correct_answer"] for r in answered), len(answered)),
        "answered_correctly_rate": rate(sum(r.get("correct_answer", False) for r in should_answer), len(should_answer)),
        "correct_abstention_rate": rate(sum(r["observed"] == "abstain" for r in should_abstain), len(should_abstain)),
        "correct_clarification_rate": rate(sum(r["observed"] == "clarify" for r in should_clarify), len(should_clarify)),
    }


def by_kind(results: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in results:
        out.setdefault(r["kind"], {}).setdefault(r["observed"], 0)
        out[r["kind"]][r["observed"]] += 1
    return out


def split_limits(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Headline results and the known-limit cases, which are reported on their own."""
    return ([r for r in results if not r["kind"].startswith("limit_")],
            [r for r in results if r["kind"].startswith("limit_")])


def run_entity_bench(dataset: str | Path = DATA_PATH, make_rag: Callable[[Settings], Any] | None = None) -> dict[str, Any]:
    from ..factory import offline

    data = json.loads(Path(dataset).read_text(encoding="utf-8"))
    runs = {name: run_config(data, flags, make_rag or offline) for name, flags in CONFIGS.items()}
    return {
        "dataset": {"version": data["version"], "seed": data["seed"], "synthetic": data["synthetic"],
                    "documents": len(data["documents"]), "questions": len(data["queries"])},
        "caveat": ("Offline stack (hashing embedder, lexical reranker and NLI, extractive generator) on synthetic data: "
                   "this measures the pipeline's logic, not quality with real models or real documents."),
        "configs": {name: {"metrics": metrics(split_limits(r)[0]), "observed_by_kind": by_kind(split_limits(r)[0]),
                           "known_limits": [{"id": x["id"], "kind": x["kind"], "expected": x["expected"],
                                             "observed": x["observed"]} for x in split_limits(r)[1]]}
                    for name, r in runs.items()},
        "failures": {name: [r for r in split_limits(rs)[0] if r["observed"] != r["expected"]
                            or r.get("correct_answer") is False] for name, rs in runs.items()},
    }
