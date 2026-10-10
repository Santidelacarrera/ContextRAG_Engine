"""Core data types shared across the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
    section: str = ""  # markdown heading path / code signature the chunk starts under
    page: int | None = None  # 1-based PDF page of ``start``
    page_breaks: tuple[int, ...] = ()  # absolute offsets inside the chunk where a further page begins

    def page_at(self, offset: int) -> int | None:
        if self.page is None:
            return None
        return self.page + sum(1 for b in self.page_breaks if b <= offset)


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
    score: float = 0.0


@dataclass(frozen=True)
class Evidence:
    """Exact coordinates of the document span that backs a claim."""

    doc_id: str
    source: str
    start: int
    end: int
    text: str
    page: int | None = None


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
    cached: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Answer":
        claims = [
            ClaimVerdict(c["claim"], c["citations"], c["score"], c["supported"],
                         Evidence(**c["evidence"]) if c.get("evidence") else None, c.get("reason", ""))
            for c in d["claims"]
        ]
        sources = []
        for s in d["sources"]:
            p = dict(s["parent"])
            p["acl"], p["page_breaks"] = tuple(p["acl"]), tuple(p.get("page_breaks", ()))
            sources.append(Context(s["index"], ParentChunk(**p), s.get("score", 0.0)))
        return cls(Status(d["status"]), d["question"], d["text"], d["confidence"], claims, sources,
                   d["attempts"], d["reason"], d.get("cached", False))
