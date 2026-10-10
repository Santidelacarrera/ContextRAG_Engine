from __future__ import annotations

import hashlib
import math
from typing import Protocol, Sequence

from .text import content_tokens, stem


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic, dependency-free feature-hashing embedder (tests / offline use)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in map(stem, content_tokens(t)):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out


class SentenceTransformerEmbedder:
    """bge-large-en / bge-large-es / bge-m3 etc. via sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-large-en-v1.5", query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model)
        self.dim = self._m.get_sentence_embedding_dimension()
        self.query_prefix = query_prefix

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._m.encode(list(texts), normalize_embeddings=True).tolist()
