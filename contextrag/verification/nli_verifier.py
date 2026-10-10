"""Self-RAG attribution layer: NLI entailment of every claim against the retrieved evidence.

Two entry points:

* ``verify_attribution(statement, context_chunks) -> float`` -- standalone score in [0, 1]:
  the *minimum over the statement's claims* of the *best entailment over evidence windows*.
  One unsupported claim therefore sinks the whole statement.
* ``NLIVerifier.verify(answer, contexts)`` -- the pipeline's strict variant: each claim must
  carry ``[n]`` citations and is checked against the cited passages only, returning
  per-claim verdicts with the exact document coordinates of the supporting span.
"""
from __future__ import annotations

import re
from typing import Protocol, Sequence

from ..core.models import ClaimVerdict, Context, Evidence
from ..core.text import content_tokens, split_sentences, stem, tokenize

DEFAULT_THRESHOLD = 0.95
MAX_WINDOW = 3  # evidence windows span 1..3 consecutive sentences


def fallback_message(threshold: float = DEFAULT_THRESHOLD) -> str:
    return (f"No se puede verificar la respuesta con un {round(threshold * 100)}% de certidumbre "
            "basada en las fuentes proporcionadas")


# ---- NLI models --------------------------------------------------------------------------

class NLIModel(Protocol):
    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each (premise, hypothesis) pair."""
        ...


_NEG = {"no", "not", "never", "none", "without", "cannot", "nunca", "sin", "tampoco", "nadie", "ningún", "ninguno"}
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def _align(h: list[str], p: list[str]) -> list[int]:
    """Indices into ``p`` of a longest common subsequence of ``h`` and ``p`` (DP with traceback)."""
    n, m = len(h), len(p)
    t = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            t[i][j] = t[i + 1][j + 1] + 1 if h[i] == p[j] else max(t[i + 1][j], t[i][j + 1])
    i = j = 0
    out = []
    while i < n and j < m:
        if h[i] == p[j]:
            out.append(j)
            i += 1
            j += 1
        elif t[i + 1][j] >= t[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


class LexicalEntailment:
    """Conservative heuristic NLI usable offline.

    Score = (longest in-order run of the hypothesis' content terms found in the premise) / (number of
    hypothesis content terms). Being order-sensitive, "Luis aprobó a Ana" is NOT entailed by "Ana aprobó a
    Luis". The score is 0 if the hypothesis introduces a number the premise lacks, or if negation differs
    between the hypothesis and the stretch of the premise it aligns to (a negation in an unrelated clause
    of the same sentence is ignored). It rejects honest paraphrase and re-ordering (fails closed) and cannot
    detect every subtle contradiction; use ``DebertaNLI`` for semantic entailment in production.
    """

    def entailment(self, pairs):
        return [self._one(p, h) for p, h in pairs]

    @staticmethod
    def _one(premise: str, hyp: str) -> float:
        h = [stem(t) for t in content_tokens(hyp)]
        if not h:
            return 0.0
        p = [stem(t) for t in content_tokens(premise)]
        if set(_NUM.findall(hyp)) - set(_NUM.findall(premise)):
            return 0.0
        idx = _align(h, p)
        if not idx:
            return 0.0
        if bool(_NEG & set(h)) != bool(_NEG & set(p[idx[0] : idx[-1] + 1])):
            return 0.0
        return len(idx) / len(h)


class DebertaNLI:
    """DeBERTa-v3 NLI cross-encoder (needs ``transformers`` + ``torch``)."""

    def __init__(self, model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model)
        self._m = AutoModelForSequenceClassification.from_pretrained(model).eval()
        self._ent = {k.lower(): v for k, v in self._m.config.label2id.items()}["entailment"]
        self.batch_size = batch_size

    def entailment(self, pairs):
        out: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self._tok([p for p, _ in batch], [h for _, h in batch], truncation=True, max_length=512,
                            padding=True, return_tensors="pt")
            with self._torch.no_grad():
                out.extend(self._m(**enc).logits.softmax(-1)[:, self._ent].tolist())
        return out


# ---- claim parsing -----------------------------------------------------------------------

_MARK = re.compile(r"\[(\d+)\]")
_MARK_THEN_END = re.compile(r"([.!?])\s*((?:\[\d+\]\s*)+)")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def parse_claims(answer: str) -> list[tuple[str, list[int]]]:
    """(claim text without markers, cited indices) per sentence. ``X. [1]`` is read as ``X [1].``"""
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


def _windows(text: str, base: int):
    """(window text, start, end) over 1..MAX_WINDOW consecutive sentences; offsets shifted by ``base``."""
    sents = split_sentences(text, base=base)
    for i in range(len(sents)):
        for n in range(1, MAX_WINDOW + 1):
            if i + n > len(sents):
                break
            grp = sents[i : i + n]
            yield " ".join(s for s, _, _ in grp), grp[0][1], grp[-1][2]


# ---- standalone score --------------------------------------------------------------------

def verify_attribution(statement: str, context_chunks: Sequence[str], nli: NLIModel | None = None) -> float:
    """Support score in [0, 1] of ``statement`` given ``context_chunks`` (citation markers ignored)."""
    nli = nli or LexicalEntailment()
    claims = [c for c, _ in parse_claims(statement)]
    if not claims or not context_chunks:
        return 0.0
    wins = [w for chunk in context_chunks for w, _, _ in _windows(chunk, 0)]
    if not wins:
        return 0.0
    worst = 1.0
    for claim in claims:
        scores = nli.entailment([(w, claim) for w in wins])
        worst = min(worst, max(scores))
    return float(worst)


# ---- pipeline verifier -------------------------------------------------------------------

class NLIVerifier:
    def __init__(self, nli: NLIModel | None = None, threshold: float = DEFAULT_THRESHOLD):
        self.nli, self.threshold = nli or LexicalEntailment(), threshold

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
            cand = [(w, s, e, by_index[i]) for i in dict.fromkeys(cites)
                    for w, s, e in _windows(by_index[i].parent.text, by_index[i].parent.start)]
            if not cand:
                verdicts.append(ClaimVerdict(claim, cites, 0.0, False, None, "cited passage empty"))
                continue
            scores = self.nli.entailment([(w, claim) for w, _, _, _ in cand])
            # highest entailment; among ties the tightest window, so evidence coordinates are minimal
            best = max(range(len(scores)), key=lambda i: (round(scores[i], 6), -len(cand[i][0])))
            w, s, e, ctx = cand[best]
            ok = scores[best] >= self.threshold
            p = ctx.parent
            verdicts.append(ClaimVerdict(
                claim, cites, float(scores[best]), ok,
                Evidence(p.doc_id, p.source, s, e, w, p.page_at(s)) if ok else None,
                "" if ok else f"best entailment {scores[best]:.2f} < {self.threshold}"))
        return verdicts
