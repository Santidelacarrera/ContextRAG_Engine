"""Retrieval-side RAG metrics that need no LLM judge and therefore run deterministically in CI.

These are not RAGAS. RAGAS' faithfulness, answer relevancy and LLM context-precision need a judge model, which this
repository cannot run offline. The metrics here use the reference answer as the relevance oracle:

* ``context_recall``: the share of the reference answer's key terms present somewhere in the evidence handed to
  the model. A ceiling on what any generator can answer from this evidence.
* ``average_precision``: over the evidence in prompt order, the mean precision at each position holding a passage
  that carries *all* the reference's key terms. It rewards putting the relevant passage first.

Both are lexical. They miss paraphrases and can be fooled by a passage that repeats the key terms out of context.
Treat them as regression signals on a fixed corpus, not as measures of answer correctness.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from ..core.models import Context
from ..core.text import content_tokens, stem


def key_terms(reference: str) -> frozenset[str]:
    return frozenset(stem(t) for t in content_tokens(reference))


def _terms_in(text: str) -> frozenset[str]:
    return frozenset(stem(t) for t in content_tokens(text))


def context_recall(reference: str, contexts: Sequence[Context]) -> float:
    need = key_terms(reference)
    if not need:
        return 1.0
    have = frozenset().union(*(_terms_in(c.parent.text) for c in contexts)) if contexts else frozenset()
    return len(need & have) / len(need)


def average_precision(reference: str, contexts: Sequence[Context]) -> float:
    need = key_terms(reference)
    if not need or not contexts:
        return 0.0 if need else 1.0
    hits, precisions = 0, []
    for position, c in enumerate(contexts, start=1):
        if need <= _terms_in(c.parent.text):  # a passage that alone carries the whole reference
            hits += 1
            precisions.append(hits / position)
    return sum(precisions) / hits if hits else 0.0


@dataclass
class ContextReport:
    total: int
    mean_recall: float
    mean_average_precision: float
    per_case: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def regressions(report: ContextReport, baseline: dict[str, float], tolerance: float = 0.0) -> list[str]:
    """Metrics that fell below the baseline by more than ``tolerance``. Empty list means no regression."""
    current = {"mean_recall": report.mean_recall, "mean_average_precision": report.mean_average_precision}
    return [f"{k}: {current[k]:.4f} < baseline {v:.4f}" for k, v in baseline.items()
            if k in current and current[k] + tolerance < v]


def evaluate_context(make_rag, data: dict, tenant: str = "eval") -> ContextReport:
    """Run retrieval and evidence assembly exactly as ``/query`` does, then score the evidence against each
    answerable case's reference. Unanswerable cases are skipped: they have no reference to measure against."""
    from ..context.budget import assemble_evidence, span_context

    rag = make_rag()
    for d in data["documents"]:
        rag.ingestor.ingest_text(d["text"], tenant=tenant, doc_id=d["doc_id"], source=d["doc_id"])
    per_case: dict[str, dict[str, float]] = {}
    for c in data["cases"]:
        if not c["answerable"]:
            continue
        detailed = rag.retriever.retrieve_detailed(c["question"], tenant)
        spans = {ctx.parent.id: span_context(ctx, child) for ctx, child in detailed}
        evidence = assemble_evidence([ctx for ctx, _ in detailed], rag.budget,
                                     dedupe_threshold=rag.cfg.context_dedupe_jaccard,
                                     placement=rag.cfg.context_placement, spans=spans)
        per_case[c["id"]] = {"recall": context_recall(c["expect"], evidence),
                             "average_precision": average_precision(c["expect"], evidence)}
    n = len(per_case)
    return ContextReport(
        total=n,
        mean_recall=sum(v["recall"] for v in per_case.values()) / n if n else 0.0,
        mean_average_precision=sum(v["average_precision"] for v in per_case.values()) / n if n else 0.0,
        per_case=per_case,
    )
