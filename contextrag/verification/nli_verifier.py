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
import unicodedata
from functools import lru_cache
from typing import Protocol, Sequence

from ..core.models import ClaimVerdict, Context, Evidence
from ..core.text import STOPWORDS, _ends_with_abbreviation, content_tokens, is_abbreviation, split_sentences, stem

DEFAULT_THRESHOLD = 0.95
MAX_WINDOW = 3  # evidence windows span 1..3 consecutive sentences
EVIDENCE_TOLERANCE = 0.02  # windows this close to the best score compete for being reported as evidence


def fallback_message(threshold: float = DEFAULT_THRESHOLD) -> str:
    return (f"No se puede verificar la respuesta con un {round(threshold * 100)}% de certidumbre "
            "basada en las fuentes proporcionadas")


# ---- NLI models --------------------------------------------------------------------------

class NLIModel(Protocol):
    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each (premise, hypothesis) pair."""
        ...


# ---- heuristic NLI -------------------------------------------------------------------------
#
# Terms whose omission changes meaning. If one is present in the premise clause a claim is aligned to and
# the claim does not carry it too, the claim is NOT entailed ("Employees may work remotely" !=> "Employees work remotely").
_NEG_WORDS = """no not never none nobody nothing neither nor without unable lack lacks lacking absent absence cannot
nunca jamás ningún ninguno ninguna nadie nada tampoco ni sin""".split()
_QUALIFIER_WORDS = """may might could can should must shall required possible possibly perhaps maybe probably likely
unlikely allegedly reportedly supposedly apparently claim claims claimed said says say stated according believed
alleged unclear unknown uncertain doubtful doubt false untrue incorrect wrong deny denies denied refuse refuses refused
reject rejects rejected prohibited forbidden banned fail fails failed only just merely solely some several few most
mostly partially partly sometimes occasionally usually often generally typically rarely seldom hardly barely except
excluding excluded besides least fewer less more than approximately nearly almost
puede pueden podría podrían podrá debe deben debería quizá quizás posible posiblemente probablemente supuestamente
presuntamente afirma afirman dijo dice según falso falsa incorrecto niega negó rechaza rechazó prohibido prohibida
solo sólo solamente algunos algunas varios varias pocos pocas mayoría parcialmente generalmente raramente apenas
excepto salvo menos más aproximadamente casi deja dejó dejar fracasa falla""".split()
# Conditions govern the whole sentence, not just their own clause ("If admin, access is granted.").
_CONDITION_WORDS = "if unless whether provided assuming until when whenever otherwise si cuando".split()
_CLAUSE_WORDS = {"and", "but", "however", "while", "whereas", "although", "though", "or",
                 "y", "e", "o", "u", "pero", "aunque", "mientras", "sino"}
_UNCERTAINTY_SCORE = 0.9  # ceiling for "terms present but not admissibly aligned": must stay below any real threshold

_TOKEN = re.compile(r"(?<!\w)-?\d+(?:[.,]\d+)*(?!\w)|\w+(?:[-.]\w+)*|[.!?]+(?=\s|$)|[,;:]|\n")
_NUMBER = re.compile(r"-?\d[\d.,]*$")
_SENT_END = re.compile(r"[.!?]+")
_CONTRACTIONS = [(r"\bcan['’]t\b", "can not"), (r"\bcannot\b", "can not"), (r"\bwon['’]t\b", "will not"),
                 (r"\bshan['’]t\b", "shall not"), (r"\bain['’]t\b", "is not"), (r"\b(\w+)n['’]t\b", r"\1 not"),
                 (r"['’]s\b", "")]


def _canon_number(tok: str) -> str:
    """-7.50 -> -7.5, 05 -> 5, 1,500 / 1.500 -> 1500, 7,5 -> 7.5 (versions like 2.1.1 are left alone)."""
    sign, body = ("-", tok[1:]) if tok.startswith("-") else ("", tok)
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", body):
        body = re.sub(r"[.,]", "", body)
    elif re.fullmatch(r"\d+,\d+", body):
        body = body.replace(",", ".")
    if re.fullmatch(r"\d+\.\d+", body):
        body = body.rstrip("0").rstrip(".") if "." in body else body
        whole, _, frac = body.partition(".")
        body = (whole.lstrip("0") or "0") + ("." + frac if frac else "")
    elif body.isdigit():
        body = body.lstrip("0") or "0"
    return sign + body


class _Clause:
    __slots__ = ("stems", "marks", "neg", "cond")

    def __init__(self):
        self.stems: list[str] = []
        self.marks: set[str] = set()  # qualifiers + negations present (as stems)
        self.neg: set[str] = set()
        self.cond: set[str] = set()


_NEG = {stem(w) for w in _NEG_WORDS}
_QUAL = {stem(w) for w in _QUALIFIER_WORDS} | _NEG
_COND = {stem(w) for w in _CONDITION_WORDS}


class _Analysis:
    def __init__(self, text: str):
        t = unicodedata.normalize("NFKC", text).casefold().replace("%", " percent ")
        for pat, rep in _CONTRACTIONS:
            t = re.sub(pat, rep, t)
        self.sentences: list[list[_Clause]] = []
        self.numbers: set[str] = set()
        self.neg: set[str] = set()
        self.stems: set[str] = set()
        self.cdelims = 0  # clause delimiters (commas, 'and', ...) -- a claim may only span as many clauses as it has
        clauses: list[_Clause] = []
        cur = _Clause()

        def close_clause():
            nonlocal cur
            if cur.stems:
                clauses.append(cur)
            cur = _Clause()

        prev = ""
        for tok in _TOKEN.findall(t):
            was_prev, prev = prev, tok
            if _SENT_END.fullmatch(tok) and is_abbreviation(was_prev):
                continue
            if tok == "\n" or _SENT_END.fullmatch(tok):
                close_clause()
                if clauses:
                    self.sentences.append(clauses)
                    clauses = []
                continue
            if tok in ",;:" or tok in _CLAUSE_WORDS:
                self.cdelims += 1
                close_clause()
                continue
            if _NUMBER.match(tok):
                tok = _canon_number(tok)
                self.numbers.add(tok)
                st = tok
            else:
                st = stem(tok) if not any(c.isdigit() for c in tok) else tok
            is_marker = st in _QUAL
            if tok in STOPWORDS and not is_marker and st not in _COND:
                continue
            cur.stems.append(st)
            self.stems.add(st)
            if st in _QUAL:
                cur.marks.add(st)
            if st in _NEG:
                cur.neg.add(st)
                self.neg.add(st)
            if st in _COND:
                cur.cond.add(st)
        close_clause()
        if clauses:
            self.sentences.append(clauses)

    @property
    def flat(self) -> list[str]:
        return [s for sent in self.sentences for c in sent for s in c.stems]


@lru_cache(maxsize=8192)
def _analyse(text: str) -> _Analysis:
    return _Analysis(text)


def _is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(x == y for y in it) for x in needle)


class LexicalEntailment:
    """Conservative, dependency-free heuristic NLI (fails closed). A claim is entailed (score 1.0) only if ALL
    of its content terms occur, in order, inside ONE sentence of the premise, spanning no more clauses
    (separated by commas / 'and' / 'but' ...) than the claim itself does, AND that stretch carries no qualifier,
    negation or condition the claim leaves out ("may", "unless", "only", "some", "doesn't", "if ...", "claims that").
    It also rejects a number or negation the premise lacks, and stitching of facts from different sentences or
    clauses. Anything else scores at most 0.9 (set-coverage based, so it can still be ranked), which is below
    every sensible threshold.

    It is a safety net for offline use and tests, not a semantic model: it rejects honest paraphrase and cannot
    reason about meaning. Use ``DebertaNLI`` in production and calibrate with ``python -m contextrag eval --sweep``.
    """

    def entailment(self, pairs):
        return [self._one(p, h) for p, h in pairs]

    @staticmethod
    def _one(premise: str, hyp: str) -> float:
        H, P = _analyse(hyp), _analyse(premise)
        h = H.flat
        if not h:
            return 0.0
        if H.numbers - P.numbers:  # a number the premise does not contain
            return 0.0
        if H.neg and not (H.neg & P.neg):  # the claim negates something the premise never negates
            return 0.0
        missing = sum(1 for t in h if t not in P.stems)
        if missing:
            return _UNCERTAINTY_SCORE * (len(h) - missing) / len(h)
        claim_marks = {t for t in h if t in _QUAL}
        claim_conds = {t for t in h if t in _COND}
        for sent in P.sentences:
            sent_cond = set().union(*(c.cond for c in sent))
            for i in range(len(sent)):
                for j in range(i, min(len(sent), i + H.cdelims + 1)):
                    span = sent[i : j + 1]
                    if not _is_subsequence(h, [s for c in span for s in c.stems]):
                        continue
                    marks = set().union(*(c.marks for c in span))
                    neg = set().union(*(c.neg for c in span))
                    if (marks - claim_marks) or (sent_cond - claim_conds) or (H.neg - neg):
                        continue
                    return 1.0
        return _UNCERTAINTY_SCORE


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
    segments, pos = [], 0
    for m in _SPLIT.finditer(normalised):
        if "\n" not in m.group() and _ends_with_abbreviation(normalised[pos : m.start()]):
            continue  # abbreviation such as "Dr." or "e.g.": not a sentence end
        segments.append(normalised[pos : m.start()])
        pos = m.end()
    segments.append(normalised[pos:])
    for seg in segments:
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


def _term_overlap(claim: str, window: str) -> float:
    """Share of the claim's content terms present in the window (used only to pick which span to report)."""
    c = {stem(t) for t in content_tokens(claim)}
    return len(c & {stem(t) for t in content_tokens(window)}) / len(c) if c else 0.0


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
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
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
            # Support is decided by the best entailment score. For the *evidence span* we prefer, among windows that
            # also pass the threshold and score within EVIDENCE_TOLERANCE of the best, the one that most resembles the
            # claim, then the tightest -- so saturated NLI scores never pick an arbitrary (e.g. heading) sentence.
            top = max(scores)
            floor = max(self.threshold, top - EVIDENCE_TOLERANCE) if top >= self.threshold else top
            eligible = [i for i in range(len(scores)) if scores[i] >= floor]
            sim = [_term_overlap(claim, cand[i][0]) for i in range(len(cand))]
            best = max(eligible, key=lambda i: (round(sim[i], 6), -len(cand[i][0]), scores[i]))
            w, s, e, ctx = cand[best]
            ok = scores[best] >= self.threshold
            p = ctx.parent
            verdicts.append(ClaimVerdict(
                claim, cites, float(scores[best]), ok,
                Evidence(p.doc_id, p.source, s, e, w, p.page_at(s)) if ok else None,
                "" if ok else f"best entailment {scores[best]:.2f} < {self.threshold}"))
        return verdicts
