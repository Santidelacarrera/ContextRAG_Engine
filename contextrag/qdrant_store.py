"""Qdrant-backed store: named dense vector + IDF-weighted sparse (BM25-style) vector.

The tenant filter is applied inside Qdrant on every query, never post-hoc.
"""
from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from typing import Sequence

from qdrant_client import QdrantClient, models as qm

from .models import ChildChunk, ParentChunk
from .security import TenantCipher
from .store import _terms

_SPARSE_BUCKETS = 2**20


def _sparse_vec(text: str) -> qm.SparseVector:
    tf: Counter[int] = Counter()
    for t in _terms(text):
        tf[int.from_bytes(hashlib.blake2b(t.encode(), digest_size=4).digest(), "big") % _SPARSE_BUCKETS] += 1
    return qm.SparseVector(indices=list(tf), values=[float(v) for v in tf.values()])


def _uid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


class QdrantStore:
    def __init__(
        self,
        dim: int,
        *,
        client: QdrantClient | None = None,
        collection: str = "contextrag",
        cipher: TenantCipher | None = None,
    ):
        self.client = client or QdrantClient(":memory:")
        self.col, self.pcol, self.cipher = collection, f"{collection}_parents", cipher
        if not self.client.collection_exists(self.col):
            self.client.create_collection(
                self.col,
                vectors_config={"dense": qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
                sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
            for f in ("tenant", "doc_id", "acl"):
                self.client.create_payload_index(self.col, f, qm.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.col, "public", qm.PayloadSchemaType.BOOL)
        if not self.client.collection_exists(self.pcol):
            self.client.create_collection(self.pcol, vectors_config=qm.VectorParams(size=1, distance=qm.Distance.DOT))
            for f in ("tenant", "doc_id"):
                self.client.create_payload_index(self.pcol, f, qm.PayloadSchemaType.KEYWORD)

    # -- text at rest ---------------------------------------------------
    def _enc(self, tenant: str, text: str) -> str:
        return self.cipher.encrypt(tenant, text) if self.cipher else text

    def _dec(self, tenant: str, text: str) -> str:
        return self.cipher.decrypt(tenant, text) if self.cipher else text

    def _filter(self, tenant: str, groups: frozenset[str]) -> qm.Filter:
        return qm.Filter(
            must=[
                qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)),
                qm.Filter(
                    should=[
                        qm.FieldCondition(key="public", match=qm.MatchValue(value=True)),
                        qm.FieldCondition(key="acl", match=qm.MatchAny(any=sorted(groups) or [""])),
                    ]
                ),
            ]
        )

    def add(self, parents: Sequence[ParentChunk], children: Sequence[ChildChunk], vectors):
        if parents:
            self.client.upsert(
                self.pcol,
                [
                    qm.PointStruct(
                        id=_uid(p.id),
                        vector=[0.0],
                        payload={
                            "pid": p.id, "tenant": p.tenant, "doc_id": p.doc_id, "source": p.source,
                            "text": self._enc(p.tenant, p.text), "start": p.start, "end": p.end, "acl": list(p.acl),
                        },
                    )
                    for p in parents
                ],
            )
        if children:
            self.client.upsert(
                self.col,
                [
                    qm.PointStruct(
                        id=_uid(c.id),
                        vector={"dense": list(v), "sparse": _sparse_vec(c.text)},
                        payload={
                            "cid": c.id, "pid": c.parent_id, "tenant": c.tenant, "doc_id": c.doc_id,
                            "text": self._enc(c.tenant, c.text), "start": c.start, "end": c.end,
                            "acl": list(c.acl), "public": not c.acl,
                        },
                    )
                    for c, v in zip(children, vectors, strict=True)
                ],
            )

    def _child(self, tenant: str, payload: dict) -> ChildChunk:
        return ChildChunk(
            payload["cid"], payload["pid"], payload["doc_id"], tenant,
            self._dec(tenant, payload["text"]), payload["start"], payload["end"], tuple(payload["acl"]),
        )

    def dense(self, vec, tenant, groups, k):
        res = self.client.query_points(
            self.col, query=list(vec), using="dense", query_filter=self._filter(tenant, groups),
            limit=k, with_payload=True,
        )
        return [(self._child(tenant, p.payload), p.score) for p in res.points]

    def sparse(self, query, tenant, groups, k):
        res = self.client.query_points(
            self.col, query=_sparse_vec(query), using="sparse", query_filter=self._filter(tenant, groups),
            limit=k, with_payload=True,
        )
        return [(self._child(tenant, p.payload), p.score) for p in res.points if p.score > 0]

    def parent(self, parent_id, tenant):
        pts = self.client.retrieve(self.pcol, [_uid(parent_id)], with_payload=True)
        if not pts or pts[0].payload["tenant"] != tenant:
            return None
        d = pts[0].payload
        return ParentChunk(
            d["pid"], d["doc_id"], tenant, d["source"], self._dec(tenant, d["text"]),
            d["start"], d["end"], tuple(d["acl"]),
        )

    def delete_document(self, tenant, doc_id):
        flt = qm.Filter(must=[
            qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)),
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
        ])
        for col in (self.col, self.pcol):
            self.client.delete(col, qm.FilterSelector(filter=flt))
