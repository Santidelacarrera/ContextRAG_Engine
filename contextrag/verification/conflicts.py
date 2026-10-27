"""Conflicts between sources: two retrieved passages that say incompatible things about the same question.

When sources disagree (two different dates for the same hearing), picking one is a decision the system has no basis to
make. The right behaviour is to say that the sources disagree, show them, and leave the decision to a person. This
module finds candidate disagreements; the pipeline then declines to answer and reports them.

Detectors (the interface is :class:`ConflictDetector`; implementations are interchangeable):

* :class:`NLIConflictDetector`: asks an NLI model whether two statements contradict each other. This is the only
  semantic detector. It needs a model whose head has a *contradiction* class, and its threshold is not calibrated
  until someone does that with labelled data. Without a real model it can only be exercised with a stand-in.
* :class:`StructuredConflictDetector`: **not semantic.** It compares dates and numbers between sentences that are
  otherwise nearly identical and are about the same named subject. It finds "the hearing is on 5 May" versus "the hearing
  is on 9 June", and nothing else: it cannot read negation, a revoked clause, or a contradiction without figures. Every
  result is a *possible* conflict that needs a human to look.
* A stand-in NLI model for tests is just any object with a ``contradiction`` method (see the tests); it exists to prove
  the plumbing, not the detection quality.

A result never says which side is right. Both sources are returned with document, page and offsets.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..core.models import Context, ParentChunk
from ..core.text import STOPWORDS, content_tokens, split_sentences
from .entities import named_phrases

MONTHS = {m: i for i, names in enumerate(
    ["enero january jan", "febrero february feb", "marzo march mar", "abril april apr", "mayo may", "junio june jun",
     "julio july jul", "agosto august aug", "septiembre setiembre september sep", "octubre october oct",
     "noviembre november nov", "diciembre december dec"], start=1) for m in names.split()}
_MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
_DATE_PATTERNS = [
    (re.compile(rf"\b(\d{{1,2}})\s+(?:de\s+)?({_MONTH_RE})\.?(?:\s+(?:de\s+|del\s+)?(\d{{4}}))?\b", re.I), "dmy"),
    (re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?(?:\s+(\d{{4}}))?\b", re.I), "mdy"),
    (re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"), "slash"),
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
]
_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d+))?(?:\s*(millones|millón|mil)\b)?", re.I)
# Uncalibrated. Stricter than a loose match on purpose: 'signed on 5 May' and 'expires on 5 May 2029' share most of their
# words and are not a contradiction. A looser threshold finds more and flags complementary facts as conflicts.
_MIN_JACCARD, _MIN_JACCARD_UNNAMED = 0.75, 0.85


@dataclass(frozen=True)
class SourceRef:
    """Where a statement comes from, precisely enough to open it."""
    doc_id: str
    source: str
    page: int | None
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "source": self.source, "page": self.page, "start": self.start, "end": self.end,
                "text": self.text[:400]}

    @property
    def unit(self) -> tuple[str, int | None]:
        return (self.doc_id, self.page)


@dataclass(frozen=True)
class Conflict:
    kind: str  # "value_mismatch" (structured) or "contradiction" (NLI)
    method: str  # which detector said so, so a reader knows how much to trust it
    score: float
    a: SourceRef
    b: SourceRef
    needs_human_review: bool = True  # always: no detector here is allowed to decide which source is right

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "method": self.method, "score": round(self.score, 4),
                "needs_human_review": self.needs_human_review, "sources": [self.a.to_dict(), self.b.to_dict()]}


class ConflictDetector(Protocol):
    name: str

    def detect(self, question: str, contexts: Sequence[Context]) -> list[Conflict]: ...


class ContradictionModel(Protocol):
    def contradiction(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


# ---- shared helpers -------------------------------------------------------------------------------------

def page_of(parent: ParentChunk, offset: int) -> int | None:
    """Page of an absolute document offset inside ``parent`` (``None`` for documents without pages)."""
    if parent.page is None:
        return None
    return parent.page + sum(1 for b in parent.page_breaks if b <= offset)


def _candidates(question: str, contexts: Sequence[Context], per_source: int) -> list[SourceRef]:
    """For each source, the sentences that most overlap the question: only those can disagree about it."""
    q = set(content_tokens(question))
    out: list[SourceRef] = []
    for ctx in contexts:
        p = ctx.parent
        scored = []
        for sentence, s, e in split_sentences(p.text, base=p.start):
            overlap = len(q & set(content_tokens(sentence)))
            if overlap:
                scored.append((overlap, SourceRef(p.doc_id, p.source, page_of(p, s), s, e, sentence)))
        out.extend(ref for _, ref in sorted(scored, key=lambda x: -x[0])[:per_source])
    return out


def _dates(text: str) -> tuple[frozenset[str], str]:
    """ISO dates in ``text`` and the text with them removed (so their digits are not counted as numbers)."""
    found: set[str] = set()
    for pattern, order in _DATE_PATTERNS:
        def take(m: re.Match[str], order: str = order) -> str:
            g = m.groups()
            try:
                if order == "dmy":
                    day, month, year = int(g[0]), MONTHS[g[1].lower()], g[2] or "????"
                elif order == "mdy":
                    month, day, year = MONTHS[g[0].lower()], int(g[1]), g[2] or "????"
                elif order == "slash":
                    day, month, year = int(g[0]), int(g[1]), g[2]
                else:
                    year, month, day = g[0], int(g[1]), int(g[2])
            except (KeyError, ValueError):
                return m.group()
            found.add(f"{year}-{month:02d}-{day:02d}")
            return " "

        text = pattern.sub(take, text)
    return frozenset(found), text


# Figures written as words. Only the common forms in contracts and rules; anything else is not read, which is safe.
_NUMBER_WORDS = {w: v for w, v in {
    "cero": 0, "uno": 1, "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50, "cien": 100, "cien mil": 100000,
}.items()}
_NEGATIONS = frozenset("no nunca jamás tampoco ni sin".split())
_WORD_NUMBER = re.compile(r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def _numbers(text: str) -> frozenset[str]:
    out = {f"{float(_NUMBER_WORDS[m.group(1).lower()]):g}" for m in _WORD_NUMBER.finditer(text)}
    for m in _NUMBER.finditer(text):
        whole, frac, scale = m.group(1).replace(".", ""), m.group(2), (m.group(3) or "").lower()
        value = float(f"{whole}.{frac}" if frac else whole) * {"millones": 1e6, "millón": 1e6, "mil": 1e3, "": 1}[scale]
        out.add(f"{value:g}")
    return frozenset(out)


def _words(text: str) -> frozenset[str]:
    return frozenset(t for t in content_tokens(text) if not t.isdigit() and t not in MONTHS and t not in STOPWORDS)


def _negated(text: str) -> bool:
    return any(t in _NEGATIONS for t in re.findall(r"\w+", text.lower()))


def _differ(a: frozenset[str], b: frozenset[str]) -> bool:
    """Each side has a value the other lacks. A superset ("on 5 May at 10:00" vs "on 5 May") is more detail, not conflict."""
    return bool(a and b and a - b and b - a)


# ---- detectors ----------------------------------------------------------------------------------------------

class StructuredConflictDetector:
    """Dates or figures that differ between near-identical statements about the same named subject. Not semantic."""

    name = "structured-values (heuristic; dates and figures only, not semantic)"

    def __init__(self, per_source: int = 3):
        self.per_source = per_source

    def detect(self, question: str, contexts: Sequence[Context]) -> list[Conflict]:
        refs = _candidates(question, contexts, self.per_source)
        found: list[Conflict] = []
        for i, a in enumerate(refs):
            for b in refs[i + 1:]:
                if a.unit == b.unit:
                    continue  # one page of one document does not contradict itself here
                score = self._score(a.text, b.text)
                if score:
                    found.append(Conflict("value_mismatch", self.name, score, a, b))
        return found

    @staticmethod
    def _score(a: str, b: str) -> float:
        dates_a, rest_a = _dates(a)
        dates_b, rest_b = _dates(b)
        negation_flip = _negated(a) != _negated(b)
        if not (_differ(dates_a, dates_b) or _differ(_numbers(rest_a), _numbers(rest_b)) or negation_flip):
            return 0.0
        wa, wb = _words(a), _words(b)
        if not wa or not wb:
            return 0.0
        jaccard = len(wa & wb) / len(wa | wb)
        if negation_flip and not (_differ(dates_a, dates_b) or _differ(_numbers(rest_a), _numbers(rest_b))):
            # the same statement with and without a negation word: a contradiction, but only when the wording is almost
            # identical, so that "no" in an unrelated clause does not turn two different facts into a conflict
            if jaccard >= _MIN_JACCARD:
                subject_a = {p.lower() for p in named_phrases(a)}
                if {p.lower() for p in named_phrases(b)} == subject_a:
                    return jaccard
            return 0.0
        subject_a = {p.lower() for p in named_phrases(a)}
        subject_b = {p.lower() for p in named_phrases(b)}
        if subject_a != subject_b:
            return 0.0  # about different things ("expediente Alfa" and "expediente Bravo")
        return jaccard if jaccard >= (_MIN_JACCARD if subject_a else _MIN_JACCARD_UNNAMED) else 0.0


class NLIConflictDetector:
    """Asks an NLI model whether two statements contradict each other. The threshold is uncalibrated by default."""

    def __init__(self, model: ContradictionModel, threshold: float = 0.9, per_source: int = 3, max_pairs: int = 64,
                 name: str | None = None):
        if not callable(getattr(model, "contradiction", None)):
            raise ValueError("the NLI model has no 'contradiction' method; use a model with a contradiction class "
                             "(for example DebertaNLI) or choose the structured detector")
        self.model, self.threshold, self.per_source, self.max_pairs = model, threshold, per_source, max_pairs
        self.name = name or f"nli:{type(model).__name__} (threshold {threshold}, uncalibrated)"

    def detect(self, question: str, contexts: Sequence[Context]) -> list[Conflict]:
        refs = _candidates(question, contexts, self.per_source)
        pairs = [(a, b) for i, a in enumerate(refs) for b in refs[i + 1:] if a.unit != b.unit][: self.max_pairs]
        if not pairs:
            return []
        forward = self.model.contradiction([(a.text, b.text) for a, b in pairs])
        backward = self.model.contradiction([(b.text, a.text) for a, b in pairs])
        return [Conflict("contradiction", self.name, max(f, r), a, b)
                for (a, b), f, r in zip(pairs, forward, backward, strict=True) if max(f, r) >= self.threshold]


def build_detector(mode: str, nli: Any, threshold: float) -> ConflictDetector | None:
    """The detector for a configuration value. ``nli`` mode refuses to start rather than silently doing nothing."""
    if mode == "off":
        return None
    if mode == "structured":
        return StructuredConflictDetector()
    if mode == "nli":
        return NLIConflictDetector(nli, threshold)
    raise ValueError(f"conflict_detection must be 'off', 'structured' or 'nli', got {mode!r}")


def ambiguous_subjects(question: str, contexts: Sequence[Context], per_source: int = 3) -> list[str]:
    """Subjects the evidence covers when the question names none.

    "When is the hearing?" asked over documents about hearings in matters Alfa and Bravo has no single answer: the
    statements have the same shape and different named subjects, and the question does not say which is meant. Returns
    the differing named subjects (empty if the question names something, or the evidence shows a single subject).

    Limit: a question that really asks for all of them ("which hearings are there?") has the same shape, so this
    is a warning by default and only blocks when the operator chooses ``clarify``.
    """
    if named_phrases(question):
        return []
    refs = _candidates(question, contexts, per_source)
    subjects: list[str] = []
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            if a.unit == b.unit:
                continue
            pa, pb = {p.lower(): p for p in named_phrases(a.text)}, {p.lower(): p for p in named_phrases(b.text)}
            if not pa or not pb or pa.keys() == pb.keys():
                continue
            drop = {w for p in (*pa, *pb) for w in content_tokens(p)}
            wa, wb = _words(a.text) - drop, _words(b.text) - drop
            if wa and wb and len(wa & wb) / len(wa | wb) >= _MIN_JACCARD:
                for name in (*pa.values(), *pb.values()):
                    if name not in subjects:
                        subjects.append(name)
    return subjects
