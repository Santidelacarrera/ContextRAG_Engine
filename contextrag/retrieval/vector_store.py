"""Dense vector stores over child chunks, plus parent-chunk lookup.

Tenant isolation and group ACLs are enforced inside the query (never post-hoc), and
``iter_children`` lets the BM25 index be rebuilt from the durable store after a restart.
"""
from __future__ import annotations

import threading
import uuid
from typing import Iterator, Protocol, Sequence

from ..core.models import ChildChunk, ParentChunk
from ..core.security import TenantCipher, visible


class VectorStore(Protocol):
    def add(self, parents: Sequence[ParentChunk], children: Sequence[ChildChunk],
            vectors: Sequence[Sequence[float]]) -> None: ...

    def dense(self, vec: Sequence[float], tenant: str, groups: frozenset[str], k: int) -> list[tuple[ChildChunk, float]]: ...

    def parent(self, parent_id: str, tenant: str) -> ParentChunk | None: ...

    def iter_children(self, tenant: str | None = None) -> Iterator[ChildChunk]: ...

    def delete_document(self, tenant: str, doc_id: str) -> None: ...

    def health(self) -> bool: ...


class InMemoryVectorStore:
    """Reference implementation: exact cosine search (vectors are expected L2-normalised). Thread-safe."""

    def __init__(self):
        self._parents: dict[str, ParentChunk] = {}
        self._children: dict[str, ChildChunk] = {}
        self._vecs: dict[str, Sequence[float]] = {}
        self._lock = threading.RLock()

    def add(self, parents, children, vectors):
        with self._lock:
            for p in parents:
                self._parents[p.id] = p
            for c, v in zip(children, vectors, strict=True):
                self._children[c.id], self._vecs[c.id] = c, v

    def dense(self, vec, tenant, groups, k):
        with self._lock:
            cands = [(c, self._vecs[c.id]) for c in self._children.values()
                     if c.tenant == tenant and visible(c.acl, groups)]
        scored = [(c, sum(a * b for a, b in zip(vec, v))) for c, v in cands]
        return sorted(scored, key=lambda x: -x[1])[:k]

    def parent(self, parent_id, tenant):
        with self._lock:
            p = self._parents.get(parent_id)
        return p if p and p.tenant == tenant else None

    def iter_children(self, tenant=None):
        with self._lock:
            return iter([c for c in self._children.values() if tenant is None or c.tenant == tenant])

    def delete_document(self, tenant, doc_id):
        with self._lock:
            for cid in [c.id for c in self._children.values() if c.tenant == tenant and c.doc_id == doc_id]:
                del self._children[cid], self._vecs[cid]
            for pid in [p.id for p in self._parents.values() if p.tenant == tenant and p.doc_id == doc_id]:
                del self._parents[pid]

    def health(self):
        return True


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
        if parents:
            self.client.upsert(self.pcol, [qm.PointStruct(
                id=_uid(p.id), vector=[0.0],
                payload={"pid": p.id, "tenant": p.tenant, "doc_id": p.doc_id, "source": p.source,
                         "text": self._enc(p.tenant, p.text), "start": p.start, "end": p.end,
                         "acl": list(p.acl), "section": p.section, "page": p.page,
                         "page_breaks": list(p.page_breaks)}) for p in parents])
        if children:
            self.client.upsert(self.col, [qm.PointStruct(
                id=_uid(c.id), vector=list(v),
                payload={"cid": c.id, "pid": c.parent_id, "tenant": c.tenant, "doc_id": c.doc_id,
                         "text": self._enc(c.tenant, c.text), "start": c.start, "end": c.end,
                         "acl": list(c.acl), "public": not c.acl})
                for c, v in zip(children, vectors, strict=True)])

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
                           tuple(d.get("page_breaks", ())))

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

    def health(self):
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
