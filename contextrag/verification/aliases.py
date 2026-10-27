"""Documented aliases: when a question names an entity by a name the documents only mention as an alias.

The entity gate abstains when the question names something the evidence never mentions. That is wrong when the documents
themselves say that the name is another way to refer to an entity they describe ("Industrias Valdivia S.A. (en adelante,
«IVSA»)"). This module reads such declarations out of retrieved text and resolves a name against them.

Rules, all deliberately conservative:

* Only an explicit declaration counts: "also known as", "también conocido como", "en adelante", "hereinafter", "a.k.a.",
  "alias". A name that merely resembles another is never treated as the same entity.
* Names are compared exactly, after Unicode normalisation, case folding and removal of a leading article. There is no
  fuzzy matching: "Halvorsen Corp" and "Halvorsen Group" are different entities.
* A name that the documents declare as an alias of two different entities is *ambiguous* and is never silently resolved.
* A name with no declaration is *unknown*.

Declarations are read only from text the caller's retrieval already returned (so access control applies before an alias
can influence anything). The patterns find declarations written the usual ways; an alias stated in an unusual form is
not found, and the question is then treated as naming an unknown entity, which is the safe direction.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

_NAME = (r"[A-ZÁÉÍÓÚÑÜ][\w&.\-]*(?:\s+(?:(?:de|del|la|las|los|y|e|&|and|of|the)\s+)?[A-ZÁÉÍÓÚÑÜ0-9][\w&.\-]*){0,5}")
_Q = r"[«\"“'‘]?"
_QE = r"[»\"”'’]?"
_LEAD = r"(?:(?:el|la|los|las|the)\s+)?"
_MARKERS = (r"(?:tambi[ée]n\s+(?:conocid[oa]|llamad[oa]|denominad[oa])\s+(?:como\s+)?|also\s+(?:known|called)\s+(?:as\s+)?|"
            r"a\.k\.a\.?\s+|alias\s+|en\s+adelante,?\s+|en\s+lo\s+sucesivo,?\s+|hereinafter,?\s+|o\s+simplemente\s+)")
_DECL = re.compile(rf"({_NAME})\s*[,(]\s*{_MARKERS}{_LEAD}{_Q}({_NAME}){_QE}", re.UNICODE)
_ARTICLE = re.compile(r"^(?:el|la|los|las|the)\s+", re.IGNORECASE)


@dataclass(frozen=True)
class AliasDecl:
    canonical: str
    alias: str


@dataclass(frozen=True)
class Resolution:
    term: str
    outcome: str  # "alias" (one documented entity) | "ambiguous" (several) | "unknown" (no declaration)
    canonicals: tuple[str, ...] = ()


def normalise(name: str) -> str:
    text = unicodedata.normalize("NFC", name).strip().strip("«»\"“”'‘’.,;:()").casefold()
    return _ARTICLE.sub("", re.sub(r"\s+", " ", text))


def extract_alias_declarations(text: str) -> list[AliasDecl]:
    found: list[AliasDecl] = []
    for m in _DECL.finditer(text):
        decl = AliasDecl(_ARTICLE.sub("", m.group(1).strip(" ,-")), _ARTICLE.sub("", m.group(2).strip(" ,-")))
        if normalise(decl.canonical) != normalise(decl.alias) and decl not in found:
            found.append(decl)
    return found


def resolve(term: str, declarations: Iterable[AliasDecl]) -> Resolution:
    key = normalise(term)
    entities: dict[str, str] = {}
    for d in declarations:
        if normalise(d.alias) == key:
            entities.setdefault(normalise(d.canonical), d.canonical)
    if not entities:
        return Resolution(term, "unknown")
    return Resolution(term, "alias" if len(entities) == 1 else "ambiguous", tuple(entities.values()))
