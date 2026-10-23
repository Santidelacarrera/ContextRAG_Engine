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

from ..core.chunking import count_tokens
from ..core.models import Context
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
) -> list[Context]:
    """Return the selected contexts in prompt order, renumbered 1..n. Empty input gives an empty list."""
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")
    if not (0.0 < dedupe_threshold <= 1.0):
        raise ValueError("dedupe_threshold must be in (0, 1]")

    ranked = sorted(candidates, key=lambda c: (-c.score, c.parent.id))
    limit = budget.effective_tokens
    selected: list[Context] = []
    kept_token_sets: list[set[str]] = []
    used = 0
    for c in ranked:
        tokens = set(tokenize(c.parent.text))
        if any(jaccard(tokens, kept) >= dedupe_threshold for kept in kept_token_sets):
            continue
        cost = token_fn(c.parent.text)
        if used + cost > limit:
            continue
        selected.append(c)
        kept_token_sets.append(tokens)
        used += cost

    ordered = edge_order(selected) if placement == "edges" else selected
    return [replace(c, index=position) for position, c in enumerate(ordered, start=1)]
