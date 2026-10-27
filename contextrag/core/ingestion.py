"""Document loading (PDF / Markdown / text / source code) and indexing."""
from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import replace
import re
from html.parser import HTMLParser
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import CODE_SUFFIXES, MARKDOWN_SUFFIXES, chunk_document, detect_kind
from .config import Settings
from .tracing import NULL_TRACER, Tracer

TEXT_SUFFIXES = {".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv"}
INGESTIBLE = TEXT_SUFFIXES | CODE_SUFFIXES | MARKDOWN_SUFFIXES | {".pdf", ".html", ".htm", ".xhtml"}


class EmptyDocumentError(ValueError):
    """The document has no extractable text (e.g. a scanned PDF); the existing index is left untouched."""


class DocumentRejected(ValueError):
    """A file was refused before or during parsing. ``code`` is stable and safe to show to a client; the message
    never contains document content or the parser's own error text.

    Codes: ``too_large``, ``signature_mismatch``, ``binary_content``, ``encrypted``, ``too_many_pages``,
    ``malformed_pdf``.
    """

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


# Leading bytes of common non-text formats. A file whose name says "text" but whose bytes say otherwise is refused.
_BINARY_MAGIC = (b"%PDF-", b"PK\x03\x04", b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"\x7fELF", b"\xd0\xcf\x11\xe0")
_SNIFF_BYTES = 8192


_SKIP = {"script", "style", "noscript", "template", "svg", "head"}
_BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "nav", "aside", "ul", "ol", "li", "table",
          "tr", "blockquote", "pre", "figure", "figcaption", "dl", "dt", "dd", "form", "fieldset", "br", "hr"}


_VOID = {"br", "hr", "img", "input", "meta", "link", "area", "base", "col", "embed", "source", "track", "wbr"}
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?:px|pt|em|%)?\s*(?:;|$)", re.I)


def _is_hidden(attrs) -> bool:
    """Elements a browser would not show: ``hidden``, ``aria-hidden="true"``, or an inline style that hides them. This
    is a content-hygiene measure against hidden instructions, not a defence on its own: text can also be hidden by an
    external stylesheet or by colour, which cannot be judged without rendering the page."""
    for name, value in attrs:
        if name == "hidden" or (name == "aria-hidden" and (value or "").lower() == "true"):
            return True
        if name == "style" and _HIDDEN_STYLE.search(value or ""):
            return True
    return False


class _HtmlText(HTMLParser):
    """Visible text only; <h1>-<h6> become markdown headings so the structural chunker keeps sections."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._title = False
        self.title = ""
        self._hidden_tag, self._hidden_depth = "", 0

    def handle_starttag(self, tag, attrs):
        if self._hidden_tag:
            if tag == self._hidden_tag:
                self._hidden_depth += 1
            return
        if tag not in _VOID and _is_hidden(attrs):  # text a reader never sees can still be read by the model
            self._hidden_tag, self._hidden_depth = tag, 1
            return
        if tag in _SKIP:
            self._skip += 1
            self._title = tag == "head"
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in _BLOCK:
            self.parts.append("\n\n" if tag in ("p", "div", "section", "article", "table", "pre", "blockquote") else "\n")

    def handle_endtag(self, tag):
        if self._hidden_tag:
            if tag == self._hidden_tag:
                self._hidden_depth -= 1
                if self._hidden_depth == 0:
                    self._hidden_tag = ""
            return
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6", "p"):
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self._skip and not self._hidden_tag:
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


def validate_upload(data: bytes, name: str, settings: Settings | None = None) -> None:
    """Refuse a file before any parser sees it: size, and a signature that agrees with the declared type."""
    cfg = settings or Settings()
    if len(data) > cfg.max_document_bytes:
        raise DocumentRejected("too_large", f"{len(data)} bytes exceeds the limit of {cfg.max_document_bytes}")
    head = data[:_SNIFF_BYTES]
    if detect_kind(name) == "pdf":
        if b"%PDF-" not in head[:1024]:
            raise DocumentRejected("signature_mismatch", "the file is named as a PDF but is not one")
        return
    if any(head.startswith(magic) for magic in _BINARY_MAGIC):
        raise DocumentRejected("signature_mismatch", "the file content does not match its text type")
    if b"\x00" in head:
        raise DocumentRejected("binary_content", "the file contains binary data")


def load_pdf(data: bytes, settings: Settings | None = None) -> LoadedDocument:
    from pypdf import PdfReader

    cfg = settings or Settings()
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):  # an empty user password is how many PDFs are "locked"
            raise DocumentRejected("encrypted", "the PDF is password-protected")
        pages = reader.pages
        if len(pages) > cfg.max_pdf_pages:
            raise DocumentRejected("too_many_pages", f"{len(pages)} pages exceeds the limit of {cfg.max_pdf_pages}")
        parts, starts, pos = [], [], 0
        for page in pages:
            starts.append(pos)
            t = (page.extract_text() or "").strip() + "\n\n"
            parts.append(t)
            pos += len(t)
    except DocumentRejected:
        raise
    except Exception as exc:  # pypdf raises many types on corrupt input; the parser's text is not passed on
        raise DocumentRejected("malformed_pdf", f"the PDF could not be read ({type(exc).__name__})") from None
    return LoadedDocument("".join(parts), "pdf", starts)


def from_text(text: str, name: str, kind: str | None = None) -> LoadedDocument:
    """Text already decoded; HTML is reduced to its visible text (offsets/evidence refer to that extracted text)."""
    kind = kind or detect_kind(name)
    if kind == "html":
        return LoadedDocument(html_to_text(text), "markdown")
    return LoadedDocument(text, "text" if kind == "pdf" else kind)


def load_bytes(data: bytes, name: str, settings: Settings | None = None) -> LoadedDocument:
    """Validate, then parse. Raises :class:`DocumentRejected` for files that must not reach the parser or the index."""
    validate_upload(data, name, settings)
    if detect_kind(name) == "pdf":
        return load_pdf(data, settings)
    return from_text(data.decode("utf-8", errors="replace"), name)


def load_base64(content_b64: str, name: str, settings: Settings | None = None) -> LoadedDocument:
    return load_bytes(base64.b64decode(content_b64, validate=True), name, settings)


def _with_model(doc_hash: str, model_id: str) -> str:
    return hashlib.sha256(f"{doc_hash}\0{model_id}".encode()).hexdigest()


class Ingestor:
    """Chunks documents and writes them to the hybrid index (dense store + BM25)."""

    def __init__(self, index, embedder, settings: Settings | None = None, cache=None):
        self.index, self.embedder, self.cfg, self.cache = index, embedder, settings or Settings(), cache
        self.tracer: Tracer = NULL_TRACER  # ContextRAG replaces it; stages are chunk, embed and index

    def ingest(self, doc: LoadedDocument, *, tenant: str, doc_id: str, source: str | None = None,
               acl: tuple[str, ...] = (), force: bool = False) -> dict:
        """Index (replacing any previous version of) a document; a no-op if it is unchanged (unless ``force``)."""
        with self.tracer.span("ingest", kind=doc.kind):
            with self.tracer.span("chunk"):
                parents, children = chunk_document(
                    doc.text, doc_id=doc_id, tenant=tenant, source=source or doc_id, kind=doc.kind, acl=acl,
                    page_starts=doc.page_starts or None, settings=self.cfg)
            if not parents:  # refuse instead of silently indexing nothing (or replacing a good version with nothing)
                raise EmptyDocumentError(f"{doc_id}: no extractable text (scanned PDF or empty file?)")
            # The fingerprint also names the embedding model. Vectors from two models live in unrelated spaces: an
            # unchanged document must be re-embedded when the model changes, or queries would compare across spaces.
            parents = [replace(p, doc_hash=_with_model(p.doc_hash, self._model_id())) for p in parents]
            info = {"doc_id": doc_id, "kind": doc.kind, "parents": len(parents), "children": len(children)}
            if parents and not force and self.index.doc_hash(tenant, doc_id) == parents[0].doc_hash:
                return {**info, "unchanged": True}  # identical content/ACL/config already indexed: no embeddings, no churn
            texts = [c.text for c in children]
            vectors: list = []
            with self.tracer.span("embed", chunks=len(texts)):
                for i in range(0, len(texts), self.cfg.embed_batch_size):
                    vectors.extend(self.embedder.embed(texts[i : i + self.cfg.embed_batch_size]))
            with self.tracer.span("index", chunks=len(children)):
                self.index.replace_document(tenant, doc_id, parents, children, vectors)
            self._invalidate(tenant)
            return {**info, "unchanged": False}

    def _model_id(self) -> str:
        return getattr(self.embedder, "model_id", None) or f"{type(self.embedder).__name__}:{self.embedder.dim}"

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
