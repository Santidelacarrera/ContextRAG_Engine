"""Document loading (PDF / Markdown / text / source code) and indexing."""
from __future__ import annotations

import base64
import io
from html.parser import HTMLParser
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import CODE_SUFFIXES, MARKDOWN_SUFFIXES, chunk_document, detect_kind
from .config import Settings

TEXT_SUFFIXES = {".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv"}
INGESTIBLE = TEXT_SUFFIXES | CODE_SUFFIXES | MARKDOWN_SUFFIXES | {".pdf", ".html", ".htm", ".xhtml"}


class EmptyDocumentError(ValueError):
    """The document has no extractable text (e.g. a scanned PDF); the existing index is left untouched."""


_SKIP = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "nav", "aside", "ul", "ol", "li", "table",
          "tr", "blockquote", "pre", "figure", "figcaption", "dl", "dt", "dd", "form", "fieldset", "br", "hr"}


class _HtmlText(HTMLParser):
    """Visible text only; <h1>-<h6> become markdown headings so the structural chunker keeps sections."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._title = False
        self.title = ""

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
            self._title = tag == "head"
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in _BLOCK:
            self.parts.append("\n\n" if tag in ("p", "div", "section", "article", "table", "pre", "blockquote") else "\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p"):
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    import re

    p = _HtmlText()
    p.feed(html)
    p.close()
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@dataclass
class LoadedDocument:
    text: str
    kind: str
    page_starts: list[int] = field(default_factory=list)  # PDF only


def load_pdf(data: bytes) -> LoadedDocument:
    from pypdf import PdfReader

    parts, starts, pos = [], [], 0
    for page in PdfReader(io.BytesIO(data)).pages:
        starts.append(pos)
        t = (page.extract_text() or "").strip() + "\n\n"
        parts.append(t)
        pos += len(t)
    return LoadedDocument("".join(parts), "pdf", starts)


def from_text(text: str, name: str, kind: str | None = None) -> LoadedDocument:
    """Text already decoded; HTML is reduced to its visible text (offsets/evidence refer to that extracted text)."""
    kind = kind or detect_kind(name)
    if kind == "html":
        return LoadedDocument(html_to_text(text), "markdown")
    return LoadedDocument(text, "text" if kind == "pdf" else kind)


def load_bytes(data: bytes, name: str) -> LoadedDocument:
    if detect_kind(name) == "pdf":
        return load_pdf(data)
    return from_text(data.decode("utf-8", errors="replace"), name)


def load_base64(content_b64: str, name: str) -> LoadedDocument:
    return load_bytes(base64.b64decode(content_b64, validate=True), name)


class Ingestor:
    """Chunks documents and writes them to the hybrid index (dense store + BM25)."""

    def __init__(self, index, embedder, settings: Settings | None = None, cache=None):
        self.index, self.embedder, self.cfg, self.cache = index, embedder, settings or Settings(), cache

    def ingest(self, doc: LoadedDocument, *, tenant: str, doc_id: str, source: str | None = None,
               acl: tuple[str, ...] = (), force: bool = False) -> dict:
        """Index (replacing any previous version of) a document; a no-op if it is unchanged (unless ``force``)."""
        parents, children = chunk_document(
            doc.text, doc_id=doc_id, tenant=tenant, source=source or doc_id, kind=doc.kind, acl=acl,
            page_starts=doc.page_starts or None, settings=self.cfg)
        if not parents:  # refuse instead of silently indexing nothing (or replacing a good version with nothing)
            raise EmptyDocumentError(f"{doc_id}: no extractable text (scanned PDF or empty file?)")
        info = {"doc_id": doc_id, "kind": doc.kind, "parents": len(parents), "children": len(children)}
        if parents and not force and self.index.doc_hash(tenant, doc_id) == parents[0].doc_hash:
            return {**info, "unchanged": True}  # identical content/ACL/config already indexed: no embeddings, no churn
        texts = [c.text for c in children]
        vectors: list = []
        for i in range(0, len(texts), self.cfg.embed_batch_size):
            vectors.extend(self.embedder.embed(texts[i : i + self.cfg.embed_batch_size]))
        self.index.replace_document(tenant, doc_id, parents, children, vectors)
        self._invalidate(tenant)
        return {**info, "unchanged": False}

    def delete(self, *, tenant: str, doc_id: str) -> None:
        self.index.delete_document(tenant, doc_id)
        self._invalidate(tenant)

    def delete_tenant(self, tenant: str) -> int:
        n = self.index.delete_tenant(tenant)
        self._invalidate(tenant)
        return n

    def _invalidate(self, tenant: str) -> None:
        if self.cache:  # cached query results for this tenant are now stale
            self.cache.incr(f"epoch:{tenant}")

    def ingest_text(self, text: str, *, tenant: str, doc_id: str, source: str | None = None, kind: str | None = None,
                    acl: tuple[str, ...] = (), force: bool = False) -> dict:
        return self.ingest(from_text(text, source or doc_id, kind), tenant=tenant,
                           doc_id=doc_id, source=source, acl=acl, force=force)

    def ingest_path(self, root: Path, *, tenant: str, acl: tuple[str, ...] = (), force: bool = False) -> list[dict]:
        root = Path(root)
        base = root.parent if root.is_file() else root
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        out = []
        for f in files:
            rel = f.relative_to(base)
            if f.suffix.lower() not in INGESTIBLE or any(part.startswith(".") for part in rel.parts):
                continue
            try:
                out.append(self.ingest(load_bytes(f.read_bytes(), f.name), tenant=tenant, doc_id=rel.as_posix(),
                                       source=rel.as_posix(), acl=acl, force=force))
            except EmptyDocumentError as e:
                out.append({"doc_id": rel.as_posix(), "skipped": str(e)})
        return out
