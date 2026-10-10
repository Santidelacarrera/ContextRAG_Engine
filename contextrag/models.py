"""Core data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class ParentChunk:
    id: str
    doc_id: str
    tenant: str
    source: str
    text: str
    start: int  # char offset of ``text`` inside the source document
    end: int
    acl: tuple[str, ...] = ()  # empty => visible to the whole tenant


@dataclass(frozen=True)
class ChildChunk:
    id: str
    parent_id: str
    doc_id: str
    tenant: str
    text: str
    start: int
    end: int
    acl: tuple[str, ...] = ()


@dataclass(frozen=True)
class Context:
    """A parent chunk handed to the LLM, addressed by a 1-based citation index."""

    index: int
    parent: ParentChunk


@dataclass(frozen=True)
class Evidence:
    """Exact coordinates of the document span that backs a claim."""

    doc_id: str
    source: str
    start: int
    end: int
    text: str


@dataclass
class ClaimVerdict:
    claim: str
    citations: list[int]
    score: float
    supported: bool
    evidence: Evidence | None = None
    reason: str = ""


class Status(str, Enum):
    VERIFIED = "verified"
    FALLBACK = "fallback"


@dataclass
class Answer:
    status: Status
    question: str
    text: str
    confidence: float
    claims: list[ClaimVerdict] = field(default_factory=list)
    sources: list[Context] = field(default_factory=list)
    attempts: int = 0
    reason: str = ""
