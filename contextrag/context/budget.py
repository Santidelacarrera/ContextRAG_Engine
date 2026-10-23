"""Assemble retrieved parents into the evidence block the LLM sees.

The retriever returns parents ranked by relevance. Handing all of them to the model is unbounded (``top_k``
parents of up to ``parent_tokens`` each) and ignores that models use material at the start and end of a prompt
more reliably than material in the middle ("lost in the middle"). This module applies, in order:

1. *Deduplication*: a parent whose token set is near-identical to an already selected, better-ranked one is dropped
   (Jaccard similarity over word tokens), so the same passage is never paid for twice.
2. *Budget*: parents are admitted greedily by rank while their token cost fits the effective budget. A parent that
   does not fit is skipped, and a smaller, lower-ranked one may still fit.
3. *Placement*: ``edges`` puts the best parent first and the second-best last, with the rest between them, least
   relevant in the middle. ``descending`` keeps rank order. The order is a pure function of the input.

Citations are renumbered after placement, so ``[n]`` in the answer always refers to the prompt position.

Whether ``edges`` improves answer quality is a hypothesis for the evaluation harness to measure, not a guarantee.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from collections.abc import Mapping

from ..core.chunking import count_tokens
from ..core.models import ChildChunk, Context, ParentChunk
from ..core.text import tokenize

PLACEMENTS = ("edges", "descending")


@dataclass(frozen=True)
class ContextBudget:
    """``evidence_tokens`` is the window for passages; ``safety_margin`` is held back because the token count here
    is an estimate (word and punctuation units), not the model's own tokenizer."""

    evidence_tokens: int
    safety_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.evidence_tokens < 1:
            raise ValueError("evidence_tokens must be >= 1")
        if not (0.0 <= self.safety_margin < 1.0):
            raise ValueError("safety_margin must be in [0, 1)")

    @property
    def effective_tokens(self) -> int:
        return math.floor(self.evidence_tokens * (1.0 - self.safety_margin))


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def span_context(parent_ctx: Context, child: ChildChunk) -> Context:
    """The best-matching child of a parent, as evidence in its own right. Its offsets are absolute in the source
    document, so attribution stays exact: ``document[start:end] == text``. Its ``id`` is the child's, so it is
    deduplicated like any parent."""
    p = parent_ctx.parent
    span = ParentChunk(id=child.id, doc_id=p.doc_id, tenant=p.tenant, source=p.source, text=child.text,
                       start=child.start, end=child.end, acl=p.acl, section=p.section,
                       page=None if p.page_breaks else p.page, page_breaks=(), doc_hash=p.doc_hash)
    return replace(parent_ctx, parent=span)


def edge_order(ranked: Sequence[Context]) -> list[Context]:
    """Ranks 1, 3, 5, ... in front; 2, 4, 6, ... reversed at the back. Top-1 first, top-2 last, weakest central."""
    return list(ranked[0::2]) + list(ranked[1::2])[::-1]


def assemble_evidence(
    candidates: Sequence[Context],
    budget: ContextBudget,
    *,
    dedupe_threshold: float = 0.8,
    placement: str = "edges",
    token_fn: Callable[[str], int] = count_tokens,
    spans: Mapping[str, Context] | None = None,
) -> list[Context]:
    """Return the selected contexts in prompt order, renumbered 1..n. Empty input gives an empty list.

    ``spans`` maps a parent id to its best-matching span. When a parent does not fit the remaining budget, its span
    is tried instead, so a long document still contributes the sentence that answers the question."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    if not (0.0 < dedupe_threshold <= 1.0):
        raise ValueError("dedupe_threshold must be in (0, 1]")

    ranked = sorted(candidates, key=lambda c: (-c.score, c.parent.doc_id, c.parent.start))  # position, not id
    limit = budget.effective_tokens
    selected: list[Context] = []
    kept_token_sets: list[set[str]] = []
    used = 0
    for c in ranked:
        choice = _first_fit(c, spans, kept_token_sets, used, limit, dedupe_threshold, token_fn)
        if choice is None:
            continue
        selected.append(choice[0])
        kept_token_sets.append(choice[1])
        used += choice[2]

    ordered = edge_order(selected) if placement == "edges" else selected
    return [replace(c, index=position) for position, c in enumerate(ordered, start=1)]


def _first_fit(c: Context, spans: Mapping[str, Context] | None, kept: list[set[str]], used: int, limit: int,
               dedupe_threshold: float, token_fn: Callable[[str], int]) -> tuple[Context, set[str], int] | None:
    """The parent itself if it fits, otherwise its span if that fits. ``None`` if neither can be admitted."""
    for candidate in (c, spans.get(c.parent.id) if spans else None):
        if candidate is None:
            continue
        tokens = set(tokenize(candidate.parent.text))
        if any(jaccard(tokens, k) >= dedupe_threshold for k in kept):
            continue
        cost = token_fn(candidate.parent.text)
        if used + cost <= limit:
            return candidate, tokens, cost
    return None
