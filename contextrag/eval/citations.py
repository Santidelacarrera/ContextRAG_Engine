"""Independent audit of an answer's citations against the source documents.

The pipeline already links each supported claim to the span of evidence behind it. This module re-checks that link from
the outside, with the original document text, so that a bug in the pipeline cannot vouch for itself. It judges the
*answer's traceability*, which is separate from whether retrieval found the right documents and from whether the claim
is true: a citation can be perfectly valid for a claim that is wrong, and a right answer can have a broken citation.

Checks, per answer:

* every cited index ``[n]`` refers to a source that was actually retrieved for this answer;
* the evidence document is one of the answer's sources (no invented document);
* the evidence text is exactly ``document[start:end]`` (no invented or altered quotation);
* the evidence lies inside the source passage it is attributed to;
* the page, when reported, is a page of that document and the evidence starts on it;
* a supported claim has evidence, and an unsupported claim has none.
"""
from __future__ import annotations

from collections.abc import Mapping

from ..core.models import Answer


def citation_problems(answer: Answer, documents: Mapping[str, str],
                      page_starts: Mapping[str, list[int]] | None = None) -> list[str]:
    """Human-readable problems with the answer's citations; empty means every citation can be traced and reproduced.

    ``documents`` maps doc_id to its full original text; ``page_starts`` maps doc_id to the offset where each page begins.
    """
    problems: list[str] = []
    sources = {s.index: s for s in answer.sources}
    source_docs = {s.parent.doc_id for s in answer.sources}
    for i, claim in enumerate(answer.claims):
        label = f"claim {i + 1}"
        for n in claim.citations:
            if n not in sources:
                problems.append(f"{label}: cites [{n}], which is not one of the {len(sources)} retrieved sources")
        e = claim.evidence
        if not claim.supported:
            if e is not None:
                problems.append(f"{label}: unsupported but carries evidence")
            continue
        if e is None:
            problems.append(f"{label}: supported but has no evidence")
            continue
        if e.doc_id not in source_docs:
            problems.append(f"{label}: evidence from {e.doc_id!r}, which is not among the retrieved sources")
            continue
        text = documents.get(e.doc_id)
        if text is None:
            problems.append(f"{label}: no original text for {e.doc_id!r} to check against")
            continue
        if not (0 <= e.start < e.end <= len(text)):
            problems.append(f"{label}: offsets {e.start}..{e.end} lie outside {e.doc_id!r}")
            continue
        if text[e.start:e.end] != e.text:
            problems.append(f"{label}: quoted text is not document[{e.start}:{e.end}]")
        inside = any(s.parent.doc_id == e.doc_id and s.parent.start <= e.start and e.end <= s.parent.end
                     for s in answer.sources)
        if not inside:
            problems.append(f"{label}: evidence lies outside every retrieved passage of {e.doc_id!r}")
        if e.page is not None:
            starts = (page_starts or {}).get(e.doc_id)
            if starts is None:
                problems.append(f"{label}: reports page {e.page} but the document has no page map")
            elif not (1 <= e.page <= len(starts)):
                problems.append(f"{label}: page {e.page} does not exist (document has {len(starts)})")
            else:
                begin = starts[e.page - 1]
                end = starts[e.page] if e.page < len(starts) else len(text) + 1
                if not (begin <= e.start < end):
                    problems.append(f"{label}: evidence at {e.start} is not on page {e.page}")
    return problems
