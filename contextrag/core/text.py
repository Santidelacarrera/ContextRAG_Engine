"""Tokenisation and sentence helpers shared by BM25, rerank and verification."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[\[\"'(¿¡]?[\wÀ-ÿ])|\n+")

STOPWORDS = frozenset(
    """a an the of to in on for and or is are was were be been it its this that these those with as by at from
    el la los las un una unos unas de del al y o en es son fue ser se su sus que con por para como lo""".split()
)


ABBREVIATIONS = frozenset("""dr dra sr sra srta mr mrs ms prof profa ing lic ud uds vs etc approx aprox inc ltd corp co fig figs sec
art arts núm num pág pp cap caps ej eg ie cf al jr st e.g i.e u.s ee.uu a.m p.m p vol""".split())


def is_abbreviation(word: str) -> bool:
    """True for tokens whose trailing period does not end a sentence ("Dr.", "e.g.", "etc.", "Inc.")."""
    w = word.lower().rstrip(".")
    return w in ABBREVIATIONS


def _ends_with_abbreviation(segment: str) -> bool:
    m = re.search(r"([\w.]+)\.$", segment.rstrip())
    return bool(m) and is_abbreviation(m.group(1))


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def stem(token: str) -> str:
    """Very light suffix stripping; numbers are left untouched."""
    if token.isdigit() or len(token) <= 4:
        return token
    for suf in ("ing", "ed", "es", "s", "ción", "ciones", "mente"):
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def split_sentences(text: str, base: int = 0) -> list[tuple[str, int, int]]:
    """Split ``text`` into (sentence, start, end) with offsets shifted by ``base``."""
    out: list[tuple[str, int, int]] = []
    pos = 0
    for m in list(_SENT_END.finditer(text)) + [None]:
        if m and "\n" not in m.group() and _ends_with_abbreviation(text[pos : m.start()]):
            continue  # "Dr. Smith", "e.g. this": the period does not end the sentence
        end = m.start() if m else len(text)
        seg = text[pos:end]
        stripped = seg.strip()
        if stripped:
            s = pos + seg.index(stripped)
            out.append((stripped, base + s, base + s + len(stripped)))
        if m:
            pos = m.end()
    return out
