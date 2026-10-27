"""Parent-child chunking measured in tokens.

A token is a word or a single punctuation mark (``\\w+|[^\\w\\s]``), a close proxy for
BPE token counts on prose and source code. Documents are first cut into *structural
segments* (markdown sections / code definitions / paragraphs), segments are packed into
parents of at most ``parent_tokens`` tokens, and each parent is windowed into children of
at most ``child_tokens`` tokens with ``child_overlap_tokens`` of overlap. Every chunk keeps
absolute character offsets into the source document.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass

from .config import Settings
from .models import ChildChunk, ParentChunk

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PARA = re.compile(r"\n[ \t]*\n")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(```|~~~)")
_CODE_BOUNDARY = re.compile(
    r"^(?:def |async def |class |function |async function |export |func |fn |pub |impl |struct |enum |"
    r"interface |trait |type |public |private |protected |static |package |module |namespace |#include|"
    r"CREATE |@)"
)

CODE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs",
                 ".rb", ".php", ".kt", ".swift", ".sh", ".sql", ".scala"}
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    section: str = ""


def detect_kind(name: str) -> str:
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".html", ".htm", ".xhtml"):
        return "html"
    if suffix in MARKDOWN_SUFFIXES:
        return "markdown"
    if suffix in CODE_SUFFIXES:
        return "code"
    return "text"


def token_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _TOKEN.finditer(text)]


def count_tokens(text: str) -> int:
    return sum(1 for _ in _TOKEN.finditer(text))


def _lines(text: str):
    pos = 0
    for line in text.splitlines(keepends=True):
        yield pos, line
        pos += len(line)


# ---- structural parsers ------------------------------------------------------------------

def parse_text(text: str) -> list[Segment]:
    segs, pos = [], 0
    for m in list(_PARA.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        if text[pos:end].strip():
            segs.append(Segment(pos, end))
        pos = m.end() if m else len(text)
    return segs


def parse_markdown(text: str) -> list[Segment]:
    """Split on blank lines, but keep fenced code blocks whole and track the heading path. Skips YAML frontmatter."""
    segs: list[Segment] = []
    path: list[tuple[int, str]] = []
    in_fence = False
    seg_start: int | None = None
    seg_section = ""
    last_end = 0

    # Skip YAML frontmatter (--- at start and end)
    lines_iter = list(_lines(text))
    start_idx = 0
    if lines_iter and lines_iter[0][1].rstrip("\r\n") == "---":
        for i in range(1, len(lines_iter)):
            if lines_iter[i][1].rstrip("\r\n") == "---":
                start_idx = i + 1
                break

    def flush(end: int):
        nonlocal seg_start
        if seg_start is not None and text[seg_start:end].strip():
            segs.append(Segment(seg_start, end, seg_section))
        seg_start = None

    for idx, (pos, line) in enumerate(lines_iter):
        if idx < start_idx:
            continue
        stripped = line.rstrip("\r\n")
        if _FENCE.match(stripped):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(stripped)
        if heading:
            flush(last_end)
            level = len(heading.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, heading.group(2).strip()))
        if not in_fence and not stripped.strip() and not heading:
            flush(last_end)
            continue
        if seg_start is None:
            seg_start, seg_section = pos, " > ".join(h for _, h in path)
        last_end = pos + len(stripped)
    flush(last_end)
    return segs


def parse_code(text: str) -> list[Segment]:
    """Cut at top-level definitions (column-0 keywords); decorators stay with their definition."""
    starts: list[int] = []
    prev_decorator = False
    for pos, line in _lines(text):
        if _CODE_BOUNDARY.match(line) and not prev_decorator:
            starts.append(pos)
        prev_decorator = line.startswith("@")
    if not starts or starts[0] > 0:
        starts.insert(0, 0)
    segs = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(text)
        body = text[s:e].rstrip()
        if body:
            sig = next((ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("@")), "")
            segs.append(Segment(s, s + len(body), sig[:120] if _CODE_BOUNDARY.match(sig) else ""))
    return segs


def parse(text: str, kind: str) -> list[Segment]:
    return {"markdown": parse_markdown, "code": parse_code}.get(kind, parse_text)(text)


# ---- chunking ----------------------------------------------------------------------------

def _id(*parts: object) -> str:
    # identifiers, not secrets: usedforsecurity=False documents that and keeps the values stable across versions
    return hashlib.sha1("|".join(map(str, parts)).encode(), usedforsecurity=False).hexdigest()[:16]


def fingerprint(text: str, kind: str, source: str, acl: tuple[str, ...], cfg: Settings) -> str:
    """Identity of an indexed document: changes if the content, metadata, ACL or chunking config change."""
    h = hashlib.sha256()
    for part in (text, kind, source, ",".join(sorted(acl)),
                 f"{cfg.parent_tokens}/{cfg.child_tokens}/{cfg.child_overlap_tokens}"):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def chunk_document(
    text: str,
    *,
    doc_id: str,
    tenant: str,
    source: str,
    kind: str = "text",
    acl: tuple[str, ...] = (),
    page_starts: list[int] | None = None,
    settings: Settings | None = None,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """``page_starts`` are the char offsets at which each PDF page begins (ascending)."""
    cfg = settings or Settings()
    if cfg.child_tokens > cfg.parent_tokens or cfg.child_overlap_tokens >= cfg.child_tokens:
        raise ValueError("require child_overlap_tokens < child_tokens <= parent_tokens")
    fp = fingerprint(text, kind, source, acl, cfg)
    toks = token_spans(text)
    if not toks:
        return [], []
    tok_starts = [s for s, _ in toks]

    # token ranges [i, j) of each segment, oversize ones pre-split at parent size
    ranges: list[tuple[int, int, str]] = []
    for seg in parse(text, kind):
        i = bisect.bisect_left(tok_starts, seg.start)
        j = bisect.bisect_left(tok_starts, seg.end)
        while j - i > cfg.parent_tokens:
            ranges.append((i, i + cfg.parent_tokens, seg.section))
            i += cfg.parent_tokens
        if j > i:
            ranges.append((i, j, seg.section))

    # greedy packing of consecutive segments into parents
    packed: list[tuple[tuple[int, int], str]] = []
    cur: tuple[tuple[int, int], str] | None = None  # ((i, j), section)

    for i, j, section in ranges:
        if cur:
            i_start, j_end = cur[0]
            if j - i_start <= cfg.parent_tokens:
                # Keep deepest section: use new section if it has more nesting
                cur_section = cur[1]
                if section.count(" > ") > cur_section.count(" > "):
                    cur = ((i_start, j), section)
                else:
                    cur = ((i_start, j), cur_section)
            else:
                packed.append(cur)
                cur = ((i, j), section)
        else:
            cur = ((i, j), section)

    if cur:
        packed.append(cur)

    # Reformat packed list for compatibility with rest of code
    packed_triples: list[tuple[int, int, str]] = [(i, j, section) for (i, j), section in packed]

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    step = cfg.child_tokens - cfg.child_overlap_tokens
    for i, j, section in packed_triples:
        ps, pe = toks[i][0], toks[j - 1][1]
        pid = _id(tenant, doc_id, ps, pe)
        page = bisect.bisect_right(page_starts, ps) if page_starts else None
        breaks = tuple(x for x in page_starts[page:] if x < pe) if page_starts else ()
        parents.append(ParentChunk(pid, doc_id, tenant, source, text[ps:pe], ps, pe, acl, section, page, breaks, fp))
        k = i
        while k < j:
            m = min(k + cfg.child_tokens, j)
            cs, ce = toks[k][0], toks[m - 1][1]
            children.append(ChildChunk(_id(pid, cs), pid, doc_id, tenant, text[cs:ce], cs, ce, acl))
            if m == j:
                break
            k += step
    return parents, children
