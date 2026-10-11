import pytest

from contextrag.core.chunking import chunk_document, count_tokens, detect_kind, parse_code, parse_markdown
from contextrag.core.config import Settings
from contextrag.core.models import ParentChunk
from contextrag.core.ingestion import load_bytes
from conftest import make_pdf

SMALL = Settings(parent_tokens=60, child_tokens=20, child_overlap_tokens=5)


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def chunk(text, **kw):
    return chunk_document(text, doc_id="d", tenant="t", source="d.txt", **kw)


def test_default_sizes_are_1024_and_256_tokens():
    cfg = Settings()
    assert (cfg.parent_tokens, cfg.child_tokens) == (1024, 256)
    text = "\n\n".join(words(300, f"p{i}x") + "." for i in range(20))
    parents, children = chunk(text)
    assert len(parents) > 1 and len(children) > len(parents)
    assert all(count_tokens(p.text) <= 1024 for p in parents)
    assert all(count_tokens(c.text) <= 256 for c in children)
    assert max(count_tokens(c.text) for c in children) == 256


def test_children_link_to_parent_and_lie_inside_it():
    text = "\n\n".join(words(40, f"a{i}") for i in range(6))
    parents, children = chunk(text, settings=SMALL)
    by_id = {p.id: p for p in parents}
    assert len({p.id for p in parents}) == len(parents) and len({c.id for c in children}) == len(children)
    for c in children:
        p = by_id[c.parent_id]
        assert p.start <= c.start and c.end <= p.end
        assert p.text[c.start - p.start : c.end - p.start] == c.text
        assert (c.doc_id, c.tenant) == (p.doc_id, p.tenant)


def test_offsets_are_exact_document_coordinates():
    text = "intro line\n\n" + words(150) + "\n\nfinal paragraph here"
    parents, children = chunk(text, settings=SMALL)
    for ch in [*parents, *children]:
        assert text[ch.start : ch.end] == ch.text


def test_children_cover_parent_with_overlap():
    text = words(70)  # single paragraph of 70 tokens -> 2 parents (60 + 10), children windows of 20 step 15
    parents, children = chunk(text, settings=SMALL)
    first = [c for c in children if c.parent_id == parents[0].id]
    assert first[0].start == parents[0].start and first[-1].end == parents[0].end
    for a, b in zip(first, first[1:]):
        assert b.start < a.end  # overlap
        assert count_tokens(text[b.start : a.end]) == 5


def test_oversize_paragraph_is_split_without_loss():
    text = words(200)
    parents, _ = chunk(text, settings=SMALL)
    assert "".join(p.text.replace(" ", "") for p in parents) == text.replace(" ", "")
    assert all(count_tokens(p.text) <= 60 for p in parents)


def test_paragraphs_are_packed_into_one_parent_when_they_fit():
    parents, children = chunk("one two.\n\nthree four.\n\nfive six.")
    assert len(parents) == 1 and parents[0].text.count("\n\n") == 2 and len(children) == 1


def test_empty_and_invalid_config():
    assert chunk("") == ([], [])
    assert chunk("  \n\n  ") == ([], [])
    with pytest.raises(ValueError):
        chunk("x", settings=Settings(parent_tokens=10, child_tokens=20))
    with pytest.raises(ValueError):
        chunk("x", settings=Settings(child_tokens=10, child_overlap_tokens=10))


def test_ids_are_deterministic_and_tenant_scoped():
    a = chunk("alpha beta gamma")
    assert a == chunk("alpha beta gamma")
    other = chunk_document("alpha beta gamma", doc_id="d", tenant="other", source="d")
    assert a[0][0].id != other[0][0].id


def test_acl_is_propagated():
    p, c = chunk_document("hello world", doc_id="d", tenant="t", source="d", acl=("legal",))
    assert p[0].acl == ("legal",) and c[0].acl == ("legal",)


MD = """# Guide

Intro text.

## Setup

Run the installer.

```bash
# not a heading

echo hi
```

## Usage

Call the API.
"""


def test_markdown_tracks_heading_path_and_keeps_fences_whole():
    segs = parse_markdown(MD)
    section_of = {k: next(s.section for s in segs if k in MD[s.start : s.end])
                  for k in ("Intro text", "installer", "echo hi", "Call the API")}
    assert section_of == {"Intro text": "Guide", "installer": "Guide > Setup", "echo hi": "Guide > Setup",
                          "Call the API": "Guide > Usage"}
    fence = next(s for s in segs if "echo hi" in MD[s.start : s.end])
    assert "# not a heading" in MD[fence.start : fence.end]  # blank line inside fence did not split
    parents, _ = chunk_document(MD, doc_id="g.md", tenant="t", source="g.md", kind="markdown", settings=Settings(parent_tokens=14, child_tokens=8, child_overlap_tokens=2))
    assert any(p.section == "Guide > Usage" and "Call the API" in p.text for p in parents)


CODE = '''import os

@decorator
def alpha(x):
    return x + 1


class Beta:
    def method(self):
        return 2

def gamma():
    pass
'''


def test_code_is_cut_at_top_level_definitions():
    segs = parse_code(CODE)
    texts = [CODE[s.start : s.end] for s in segs]
    assert texts[0].startswith("import os")
    assert any(t.startswith("@decorator\ndef alpha") for t in texts)
    assert any(t.startswith("class Beta") and "def method" in t for t in texts)
    assert any(t.startswith("def gamma") for t in texts)
    assert [s.section for s in segs if "alpha" in CODE[s.start : s.end]] == ["def alpha(x):"]


def test_code_parents_do_not_cut_inside_a_small_function():
    parents, _ = chunk_document(CODE, doc_id="a.py", tenant="t", source="a.py", kind="code",
                                settings=Settings(parent_tokens=22, child_tokens=10, child_overlap_tokens=2))
    for p in parents:
        assert ("def method" in p.text) == ("return 2" in p.text)


def test_detect_kind():
    assert [detect_kind(n) for n in ("a.md", "b.PY", "c.pdf", "d.txt", "e")] == ["markdown", "code", "pdf", "text", "text"]


def test_pdf_pages_are_tracked():
    doc = load_bytes(make_pdf(["First page about apples.", "Second page about oranges."]), "f.pdf")
    assert doc.kind == "pdf" and len(doc.page_starts) == 2
    parents, _ = chunk_document(doc.text, doc_id="f.pdf", tenant="t", source="f.pdf", kind="pdf",
                                page_starts=doc.page_starts, settings=Settings(parent_tokens=6, child_tokens=4, child_overlap_tokens=1))
    pages = {("oranges" in p.text): p.page for p in parents}
    assert pages[False] == 1 and pages[True] == 2


def test_parent_spanning_pages_resolves_page_per_offset():
    doc = load_bytes(make_pdf(["Alpha beta gamma.", "Delta epsilon zeta."]), "f.pdf")
    parents, _ = chunk_document(doc.text, doc_id="f.pdf", tenant="t", source="f.pdf", kind="pdf", page_starts=doc.page_starts)
    [p] = parents  # both pages fit one parent
    assert p.page == 1 and len(p.page_breaks) == 1
    assert p.page_at(doc.text.index("Alpha")) == 1 and p.page_at(doc.text.index("Delta")) == 2
    assert ParentChunk("i", "d", "t", "s", "x", 0, 1).page_at(0) is None
