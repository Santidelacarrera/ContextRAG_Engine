"""Claim-level attribution check (the Self-RAG verification layer).

The answer is split into claims (sentences). Each claim must cite passages; its best
evidence window (1-3 consecutive sentences) inside the cited passages must be entailed
by NLI with probability >= threshold. The answer's confidence is the minimum claim score.
"""
from __future__ import annotations

import re
from typing import Sequence

from .models import ClaimVerdict, Context, Evidence
from .nli import NLIModel
from .text import split_sentences

_MARK = re.compile(r"\[(\d+)\]")
_MARK_THEN_END = re.compile(r"([.!?])\s*((?:\[\d+\]\s*)+)")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
MAX_WINDOW = 3


def parse_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Return (claim text without markers, cited indices) per sentence."""
    normalised = _MARK_THEN_END.sub(lambda m: m.group(2).strip() + m.group(1) + " ", answer)
    claims = []
    for seg in _SPLIT.split(normalised):
        seg = seg.strip()
        if not seg:
            continue
        cites = [int(x) for x in _MARK.findall(seg)]
        text = re.sub(r"\s+([.!?])", r"\1", re.sub(r"\s+", " ", _MARK.sub("", seg))).strip()
        if re.search(r"\w", text):
            claims.append((text, cites))
    return claims


def _windows(ctx: Context):
    sents = split_sentences(ctx.parent.text, base=ctx.parent.start)
    for i in range(len(sents)):
        for n in range(1, MAX_WINDOW + 1):
            if i + n > len(sents):
                break
            grp = sents[i : i + n]
            yield " ".join(s for s, _, _ in grp), grp[0][1], grp[-1][2]


class Verifier:
    def __init__(self, nli: NLIModel, threshold: float = 0.95):
        self.nli, self.threshold = nli, threshold

    def verify(self, answer: str, contexts: Sequence[Context]) -> list[ClaimVerdict]:
        by_index = {c.index: c for c in contexts}
        verdicts: list[ClaimVerdict] = []
        for claim, cites in parse_claims(answer):
            if not cites:
                verdicts.append(ClaimVerdict(claim, [], 0.0, False, None, "no citation"))
                continue
            bad = [i for i in cites if i not in by_index]
            if bad:
                verdicts.append(ClaimVerdict(claim, cites, 0.0, False, None, f"unknown citation {bad}"))
                continue
            cand: list[tuple[str, int, int, Context]] = [
                (w, s, e, by_index[i]) for i in cites for w, s, e in _windows(by_index[i])
            ]
            scores = self.nli.entailment([(w, claim) for w, _, _, _ in cand]) if cand else []
            if not scores:
                verdicts.append(ClaimVerdict(claim, cites, 0.0, False, None, "cited passage empty"))
                continue
            best = max(range(len(scores)), key=scores.__getitem__)
            w, s, e, ctx = cand[best]
            ok = scores[best] >= self.threshold
            verdicts.append(ClaimVerdict(
                claim, cites, float(scores[best]), ok,
                Evidence(ctx.parent.doc_id, ctx.parent.source, s, e, w) if ok else None,
                "" if ok else f"best entailment {scores[best]:.2f} < {self.threshold}",
            ))
        return verdicts
