"""Does the evidence mention what the question is about?

NLI answers "is this claim supported by this passage". It does not answer "is this passage about what was asked": asked
about a service that is not indexed, a system can return a faithful statement about a different service. This module is
the cheap, deterministic guard for that case. It extracts the *named terms* of a question and reports those that appear
nowhere in the evidence.

A named term is a surface form the asker chose deliberately:

* a capitalised word that is not the first word of a sentence ("... for Halxanx logs?"), or
* an identifier containing a digit ("AES-256", "ISO27001", "v2.1").

Lowercase words are never named terms, so paraphrases and synonyms ("cipher" for "algorithm") cannot trigger it.

Known limits, stated plainly. It works on surface forms: an alias the documents use instead of the name in the question
("the ticket system" for "Jira") makes it abstain, and a question that names nothing is never checked, so a wrong
attribute or a wrong relation is not caught here. It reduces one specific failure; it does not make answers correct.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

from ..core.text import STOPWORDS, tokenize

_WORD = re.compile(r"[^\W_][\w\-.]*", re.UNICODE)
_SENTENCE_BREAK = re.compile(r"[.!?¿¡:;\n]\s*$")
_ENDS_SENTENCE = re.compile(r"[.!?:;]$")


def _is_identifier(word: str) -> bool:
    return any(ch.isdigit() for ch in word) and any(ch.isalpha() for ch in word)


# Inside a name ("Código de Procedimiento Civil") versus between two names ("Alfa y Beta": coordination, two entities).
_CONNECTORS = frozenset("de del la las los of the".split())


def _classify(question: str) -> list[tuple[str, bool]]:
    """(word, is_named) for every word of the question; connectors are kept so phrases can span them."""
    out: list[tuple[str, bool]] = []
    prev_end, prev_raw = 0, ""
    for i, m in enumerate(_WORD.finditer(question)):
        raw = m.group()
        word = raw.strip(".-")
        before = question[prev_end:m.start()]
        # the previous word may carry its own full stop ("problem."), which the pattern keeps as part of the word
        sentence_initial = i == 0 or bool(_SENTENCE_BREAK.search(before)) or bool(_ENDS_SENTENCE.search(prev_raw))
        prev_end, prev_raw = m.end(), raw
        if not word:
            continue
        named = word.lower() not in STOPWORDS and ((word[0].isupper() and not sentence_initial) or _is_identifier(word))
        out.append((word, named))
    return out


def named_terms(question: str) -> list[str]:
    """Named terms of ``question`` in order of appearance, without duplicates."""
    terms: list[str] = []
    for word, named in _classify(question):
        if named and word not in terms:
            terms.append(word)
    return terms


def named_phrases(question: str) -> list[str]:
    """Named terms joined into entity phrases: "Industrias Valdivia S.A" is one phrase, not three words. A connector
    ("de", "y") joins two named words only when it sits between them."""
    phrases: list[str] = []
    run: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if run and " ".join(run) not in phrases:
            phrases.append(" ".join(run))
        run.clear()
        pending.clear()

    for word, named in _classify(question):
        if named:
            run.extend(pending)
            pending.clear()
            run.append(word)
        elif run and word.lower() in _CONNECTORS:
            pending.append(word)
        else:
            flush()
    flush()
    return phrases


def _variants(token: str) -> set[str]:
    """The token and its plural/singular forms. Deliberately not a stemmer: "halxan" must not match "halxanx"."""
    out = {token, token + "s", token + "es"}
    if token.endswith("ies") and len(token) > 4:
        out.add(token[:-3] + "y")
    if token.endswith("es") and len(token) > 4:
        out.add(token[:-2])
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        out.add(token[:-1])
    return out


def absent_terms(terms: Iterable[str], evidence: Iterable[str]) -> list[str]:
    """Those of ``terms`` with no trace in the evidence (every part of a multi-word term must be present)."""
    present = set(tokenize(" ".join(evidence)))
    missing = []
    for term in terms:
        parts = [t for t in tokenize(term) if t not in STOPWORDS]
        if parts and not all(_variants(p) & present for p in parts):
            missing.append(term)
    return missing


def missing_terms(question: str, evidence: Iterable[str]) -> list[str]:
    """Named terms of the question with no trace in the evidence."""
    return absent_terms(named_terms(question), evidence)
