"""Document loading and indexing into a tenant-isolated store."""
from __future__ import annotations

from pathlib import Path

from .chunking import chunk_document
from .config import Settings
from .embeddings import Embedder
from .store import ChunkStore

TEXT_SUFFIXES = {".md", ".txt", ".rst", ".py", ".js", ".ts", ".go", ".java", ".rs", ".c", ".h", ".cpp",
                 ".json", ".yaml", ".yml", ".toml", ".sql", ".sh"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


class Ingestor:
    def __init__(self, store: ChunkStore, embedder: Embedder, settings: Settings | None = None):
        self.store, self.embedder, self.cfg = store, embedder, settings or Settings()

    def ingest_text(self, text: str, *, tenant: str, doc_id: str, source: str | None = None,
                    acl: tuple[str, ...] = ()) -> int:
        """Index (replacing any previous version of) a document. Returns the number of child chunks."""
        self.store.delete_document(tenant, doc_id)
        parents, children = chunk_document(text, doc_id=doc_id, tenant=tenant, source=source or doc_id,
                                           acl=acl, settings=self.cfg)
        vectors = self.embedder.embed([c.text for c in children]) if children else []
        self.store.add(parents, children, vectors)
        return len(children)

    def ingest_path(self, root: Path, *, tenant: str, acl: tuple[str, ...] = ()) -> dict[str, int]:
        root = Path(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        base = root.parent if root.is_file() else root
        out = {}
        for f in files:
            if f.suffix.lower() not in TEXT_SUFFIXES | {".pdf"} or any(part.startswith(".") for part in f.relative_to(base).parts):
                continue
            rel = f.relative_to(base).as_posix()
            out[rel] = self.ingest_text(read_file(f), tenant=tenant, doc_id=rel, source=rel, acl=acl)
        return out
