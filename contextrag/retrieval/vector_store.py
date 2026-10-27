"""Dense vector stores over child chunks, plus parent-chunk lookup.

Tenant isolation and group ACLs are enforced inside the query (never post-hoc), and
``iter_children`` lets the BM25 index be rebuilt from the durable store after a restart.
"""
from __future__ import annotations

import threading
import uuid
from typing import Iterator, Protocol, Sequence

import numpy as np

from ..core.models import ChildChunk, ParentChunk
from ..core.security import TenantCipher, visible


class DimensionMismatch(ValueError):
    """Vectors from a different embedding model (or size) were offered to an index that holds another."""


class VectorStore(Protocol):
    def add(self, parents: Sequence[ParentChunk], children: Sequence[ChildChunk],
            vectors: Sequence[Sequence[float]]) -> None: ...

    def dense(self, vec: Sequence[float], tenant: str, groups: frozenset[str], k: int) -> list[tuple[ChildChunk, float]]: ...

    def parent(self, parent_id: str, tenant: str) -> ParentChunk | None: ...

    def iter_children(self, tenant: str | None = None) -> Iterator[ChildChunk]: ...

    def delete_document(self, tenant: str, doc_id: str) -> None: ...

    def doc_hash(self, tenant: str, doc_id: str) -> str | None:
        """Fingerprint of the currently indexed version of a document, if any."""
        ...

    def doc_acl(self, tenant: str, doc_id: str) -> tuple[str, ...] | None:
        """ACL of the currently indexed version of a document; ``None`` if the document is not indexed."""
        ...

    def delete_tenant(self, tenant: str) -> int:
        """Remove every chunk and vector of a tenant. Returns the number of documents removed."""
        ...

    def list_documents(self, tenant: str) -> list[dict]:
        """One summary per indexed document: doc_id, source, parents, chars, acl, doc_hash."""
        ...

    def health(self) -> bool: ...


class InMemoryVectorStore:
    """Reference implementation: exact cosine search with one numpy matrix per tenant (vectors are expected
    L2-normalised). Thread-safe. For large corpora use Qdrant (approximate search, on disk, horizontally scalable)."""

    def __init__(self):
        self._dim: int | None = None  # fixed by the first write: a later write with another size is refused
        self._parents: dict[str, ParentChunk] = {}
        self._children: dict[str, ChildChunk] = {}
        self._vecs: dict[str, Sequence[float]] = {}
        self._mats: dict[str, tuple[list[ChildChunk], np.ndarray, np.ndarray, list[int]]] = {}
        self._lock = threading.RLock()

    def add(self, parents, children, vectors):
        dims = {len(v) for v in vectors}
        if len(dims) > 1:
            raise DimensionMismatch(f"vectors of different sizes in one write: {sorted(dims)}")
        with self._lock:
            if dims and self._dim is not None and dims != {self._dim}:
                raise DimensionMismatch(f"index holds {self._dim}-dimensional vectors; got {sorted(dims)} (the embedding "
                                        "model changed: re-ingest the documents or erase the tenant)")
            if dims:
                self._dim = next(iter(dims))
            for p in parents:
                self._parents[p.id] = p
            for c, v in zip(children, vectors, strict=True):
                self._children[c.id], self._vecs[c.id] = c, v
                self._mats.pop(c.tenant, None)

    def _matrix(self, tenant):
        """(chunks, matrix, public-mask, indices of ACL-restricted rows); rebuilt lazily after a change."""
        m = self._mats.get(tenant)
        if m is None:
            chunks = [c for c in self._children.values() if c.tenant == tenant]
            if not chunks:
                return chunks, np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=bool), []
            mat = np.asarray([self._vecs[c.id] for c in chunks], dtype=np.float32)
            public = np.asarray([not c.acl for c in chunks], dtype=bool)
            m = self._mats[tenant] = (chunks, mat, public, [i for i, c in enumerate(chunks) if c.acl])
        return m

    def dense(self, vec, tenant, groups, k):
        with self._lock:
            chunks, mat, public, restricted = self._matrix(tenant)
        if not chunks:
            return []
        scores = mat @ np.asarray(vec, dtype=np.float32)
        mask = public.copy()
        for i in restricted:
            if visible(chunks[i].acl, groups):
                mask[i] = True
        idx = np.flatnonzero(mask)
        if idx.size > k:
            idx = idx[np.argpartition(-scores[idx], k - 1)[:k]]
        return [(chunks[i], float(scores[i])) for i in sorted(idx, key=lambda i: -scores[i])]

    def parent(self, parent_id, tenant):
        with self._lock:
            p = self._parents.get(parent_id)
        return p if p and p.tenant == tenant else None

    def iter_children(self, tenant=None):
        with self._lock:
            return iter([c for c in self._children.values() if tenant is None or c.tenant == tenant])

    def delete_document(self, tenant, doc_id):
        with self._lock:
            self._mats.pop(tenant, None)
            for cid in [c.id for c in self._children.values() if c.tenant == tenant and c.doc_id == doc_id]:
                del self._children[cid], self._vecs[cid]
            for pid in [p.id for p in self._parents.values() if p.tenant == tenant and p.doc_id == doc_id]:
                del self._parents[pid]

    def delete_tenant(self, tenant):
        with self._lock:
            n = len({p.doc_id for p in self._parents.values() if p.tenant == tenant})
            for cid in [c.id for c in self._children.values() if c.tenant == tenant]:
                del self._children[cid], self._vecs[cid]
            for pid in [p.id for p in self._parents.values() if p.tenant == tenant]:
                del self._parents[pid]
            self._mats.pop(tenant, None)
            return n

    def list_documents(self, tenant):
        with self._lock:
            return _summarise(p for p in self._parents.values() if p.tenant == tenant)

    def doc_hash(self, tenant, doc_id):
        with self._lock:
            return next((p.doc_hash for p in self._parents.values() if p.tenant == tenant and p.doc_id == doc_id), None)

    def doc_acl(self, tenant, doc_id):
        with self._lock:
            return next((p.acl for p in self._parents.values() if p.tenant == tenant and p.doc_id == doc_id), None)

    def health(self):
        return True


def _summarise(parents) -> list[dict]:
    docs: dict[str, dict] = {}
    for p in parents:
        d = docs.setdefault(p.doc_id, {"doc_id": p.doc_id, "source": p.source, "parents": 0, "chars": 0,
                                        "acl": list(p.acl), "doc_hash": p.doc_hash})
        d["parents"] += 1
        d["chars"] = max(d["chars"], p.end)
    return sorted(docs.values(), key=lambda d: d["doc_id"])


def _uid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


class QdrantVectorStore:
    """Qdrant-backed store (cosine). Pass ``client=QdrantClient(url=...)`` for a server,
    or omit it for Qdrant's embedded in-memory mode. Chunk text may be encrypted at rest
    with a per-tenant key (vectors stay searchable and therefore unencrypted)."""

    def __init__(self, dim: int, *, client=None, collection: str = "contextrag", cipher: TenantCipher | None = None):
        from qdrant_client import QdrantClient, models as qm

        self.qm = qm
        self.client = client or QdrantClient(":memory:")
        self.col, self.pcol, self.cipher = collection, f"{collection}_parents", cipher
        if not self.client.collection_exists(self.col):
            self.client.create_collection(self.col, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))
            for f in ("tenant", "doc_id", "acl"):
                self.client.create_payload_index(self.col, f, qm.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.col, "public", qm.PayloadSchemaType.BOOL)
        if not self.client.collection_exists(self.pcol):
            self.client.create_collection(self.pcol, vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
            for f in ("tenant", "doc_id"):
                self.client.create_payload_index(self.pcol, f, qm.PayloadSchemaType.KEYWORD)

    def _enc(self, tenant, text):
        return self.cipher.encrypt(tenant, text) if self.cipher else text

    def _dec(self, tenant, text):
        return self.cipher.decrypt(tenant, text) if self.cipher else text

    def _tenant_filter(self, tenant, groups=None):
        qm = self.qm
        must = [qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant))]
        if groups is not None:
            must.append(qm.Filter(should=[
                qm.FieldCondition(key="public", match=qm.MatchValue(value=True)),
                qm.FieldCondition(key="acl", match=qm.MatchAny(any=sorted(groups) or [""])),
            ]))
        return qm.Filter(must=must)

    def add(self, parents, children, vectors):
        qm = self.qm
        # children first, parents last: a parent carrying ``doc_hash`` is the commit marker, so an ingest that
        # died half-way is never mistaken for a completed one by the unchanged-document check.
        if children:
            self.client.upsert(self.col, [qm.PointStruct(
                id=_uid(c.id), vector=list(v),
                payload={"cid": c.id, "pid": c.parent_id, "tenant": c.tenant, "doc_id": c.doc_id,
                         "text": self._enc(c.tenant, c.text), "start": c.start, "end": c.end,
                         "acl": list(c.acl), "public": not c.acl})
                for c, v in zip(children, vectors, strict=True)])
        if parents:
            self.client.upsert(self.pcol, [qm.PointStruct(
                id=_uid(p.id), vector=[0.0],
                payload={"pid": p.id, "tenant": p.tenant, "doc_id": p.doc_id, "source": p.source,
                         "text": self._enc(p.tenant, p.text), "start": p.start, "end": p.end,
                         "acl": list(p.acl), "section": p.section, "page": p.page,
                         "page_breaks": list(p.page_breaks), "doc_hash": p.doc_hash}) for p in parents])

    def rekey(self, *, dry_run: bool = False, page: int = 256) -> dict[str, int]:
        """Re-encrypt chunk text still under a previous master key. Each point is rewritten only after its text decrypted
        and the new value decrypted back to the same plaintext; the cipher keeps reading under previous keys, so an
        interrupted run loses nothing and can be repeated. Returns the points rewritten (or that would be) per collection."""
        if self.cipher is None:
            raise RuntimeError("rekey needs a TenantCipher (MASTER_KEY)")
        counts = {"children": 0, "parents": 0}
        for name, collection in (("children", self.col), ("parents", self.pcol)):
            offset = None
            while True:
                points, offset = self.client.scroll(collection, limit=page, offset=offset, with_payload=True)
                for pt in points:
                    if not pt.payload or "tenant" not in pt.payload or "text" not in pt.payload:
                        # Abort, do not skip: a point that cannot be read would otherwise stay under the old key unnoticed.
                        # Points rewritten before this one remain rewritten (they are under the primary key); repeating the
                        # run after the point is repaired or removed completes the job. Only the collection and id are named.
                        raise RuntimeError(f"rekey aborted: {name} point {pt.id} has no tenant/text payload; "
                                           "points before it were rewritten, re-run after repairing or removing it")
                    tenant, token = pt.payload["tenant"], pt.payload["text"]
                    plaintext = self.cipher.decrypt(tenant, token)  # raises for text no configured key opens
                    if not self.cipher.needs_rotation(tenant, token):
                        continue
                    counts[name] += 1
                    if dry_run:
                        continue
                    fresh = self.cipher.encrypt(tenant, plaintext)
                    if self.cipher.decrypt(tenant, fresh) != plaintext:
                        raise RuntimeError(f"re-encryption of a {name[:-1]} did not round-trip; nothing was changed")
                    self.client.set_payload(collection, payload={"text": fresh}, points=[pt.id])
                if offset is None:
                    break
        return counts

    def _child(self, tenant, d):
        return ChildChunk(d["cid"], d["pid"], d["doc_id"], tenant, self._dec(tenant, d["text"]),
                          d["start"], d["end"], tuple(d["acl"]))

    def dense(self, vec, tenant, groups, k):
        res = self.client.query_points(self.col, query=list(vec), query_filter=self._tenant_filter(tenant, groups),
                                       limit=k, with_payload=True)
        return [(self._child(tenant, p.payload), p.score) for p in res.points]

    def parent(self, parent_id, tenant):
        pts = self.client.retrieve(self.pcol, [_uid(parent_id)], with_payload=True)
        if not pts or pts[0].payload["tenant"] != tenant:
            return None
        d = pts[0].payload
        return ParentChunk(d["pid"], d["doc_id"], tenant, d["source"], self._dec(tenant, d["text"]),
                           d["start"], d["end"], tuple(d["acl"]), d.get("section", ""), d.get("page"),
                           tuple(d.get("page_breaks", ())), d.get("doc_hash", ""))

    def iter_children(self, tenant=None):
        flt = self._tenant_filter(tenant) if tenant else None
        offset = None
        while True:
            pts, offset = self.client.scroll(self.col, scroll_filter=flt, limit=256, offset=offset, with_payload=True)
            for p in pts:
                yield self._child(p.payload["tenant"], p.payload)
            if offset is None:
                return

    def delete_document(self, tenant, doc_id):
        qm = self.qm
        flt = qm.Filter(must=[qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)),
                              qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
        for col in (self.col, self.pcol):
            self.client.delete(col, qm.FilterSelector(filter=flt))

    def delete_tenant(self, tenant):
        qm = self.qm
        n = len(self.list_documents(tenant))
        flt = qm.Filter(must=[qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant))])
        for col in (self.col, self.pcol):
            self.client.delete(col, qm.FilterSelector(filter=flt))
        return n

    def list_documents(self, tenant):
        offset, parents = None, []
        flt = self._tenant_filter(tenant)
        while True:
            pts, offset = self.client.scroll(self.pcol, scroll_filter=flt, limit=256, offset=offset, with_payload=True)
            for pt in pts:
                d = pt.payload
                parents.append(ParentChunk(d["pid"], d["doc_id"], tenant, d["source"], "", d["start"], d["end"],
                                           tuple(d["acl"]), doc_hash=d.get("doc_hash", "")))
            if offset is None:
                return _summarise(parents)

    def doc_hash(self, tenant, doc_id):
        qm = self.qm
        pts, _ = self.client.scroll(
            self.pcol, limit=1, with_payload=["doc_hash"],
            scroll_filter=qm.Filter(must=[qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)),
                                          qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]))
        return pts[0].payload.get("doc_hash") if pts else None

    def doc_acl(self, tenant, doc_id):
        qm = self.qm
        pts, _ = self.client.scroll(
            self.pcol, limit=1, with_payload=["acl"],
            scroll_filter=qm.Filter(must=[qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)),
                                          qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]))
        return tuple(pts[0].payload.get("acl", ())) if pts else None

    def health(self):
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
