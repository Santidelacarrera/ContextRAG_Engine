"""Parent-child chunking.

Parents are paragraph-aligned blocks (what the LLM reads); children are small
overlapping windows inside a parent (what the vector index matches on).
"""
from __future__ import annotations

import hashlib
import re

from .config import Settings
from .models import ChildChunk, ParentChunk

_PARA = re.compile(r"\n\s*\n")


def _id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _paragraphs(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in list(_PARA.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        if text[pos:end].strip():
            spans.append((pos, end))
        pos = m.end() if m else len(text)
    return spans


def chunk_document(
    text: str,
    *,
    doc_id: str,
    tenant: str,
    source: str,
    acl: tuple[str, ...] = (),
    settings: Settings | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    cfg = settings or Settings()
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    # Greedily pack paragraphs into parents; hard-split any oversized paragraph.
    blocks: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for s, e in _paragraphs(text):
        while e - s > cfg.parent_size:
            if cur:
                blocks.append(cur)
                cur = None
            blocks.append((s, s + cfg.parent_size))
            s += cfg.parent_size
        if cur and e - cur[0] > cfg.parent_size:
            blocks.append(cur)
            cur = None
        cur = (cur[0], e) if cur else (s, e)
    if cur:
        blocks.append(cur)

    step = max(1, cfg.child_size - cfg.child_overlap)
    for ps, pe in blocks:
        ptext = text[ps:pe]
        pid = _id(tenant, doc_id, ps, pe)
        parents.append(ParentChunk(pid, doc_id, tenant, source, ptext, ps, pe, acl))
        off = 0
        while off < len(ptext):
            piece = ptext[off : off + cfg.child_size]
            if piece.strip():
                cs = ps + off
                children.append(
                    ChildChunk(_id(pid, cs), pid, doc_id, tenant, piece, cs, cs + len(piece), acl)
                )
            if off + cfg.child_size >= len(ptext):
                break
            off += step
    return parents, children
