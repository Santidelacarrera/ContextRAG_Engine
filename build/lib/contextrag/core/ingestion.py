"""Document loading (PDF / Markdown / text / source code) and indexing."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import CODE_SUFFIXES, MARKDOWN_SUFFIXES, chunk_document, detect_kind
from .config import Settings

TEXT_SUFFIXES = {".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".csv"}
INGESTIBLE = TEXT_SUFFIXES | CODE_SUFFIXES | MARKDOWN_SUFFIXES | {".pdf"}


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


def load_bytes(data: bytes, name: str) -> LoadedDocument:
    kind = detect_kind(name)
    if kind == "pdf":
        return load_pdf(data)
    return LoadedDocument(data.decode("utf-8", errors="replace"), kind)


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

    def _invalidate(self, tenant: str) -> None:
        if self.cache:  # cached query results for this tenant are now stale
            self.cache.incr(f"epoch:{tenant}")

    def ingest_text(self, text: str, *, tenant: str, doc_id: str, source: str | None = None, kind: str | None = None,
                    acl: tuple[str, ...] = (), force: bool = False) -> dict:
        return self.ingest(LoadedDocument(text, kind or detect_kind(source or doc_id)), tenant=tenant,
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
            out.append(self.ingest(load_bytes(f.read_bytes(), f.name), tenant=tenant, doc_id=rel.as_posix(),
                                   source=rel.as_posix(), acl=acl, force=force))
        return out
