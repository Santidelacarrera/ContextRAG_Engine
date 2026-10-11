from __future__ import annotations

import os
from dataclasses import dataclass, fields


@dataclass
class Settings:
    # chunking (sizes are in tokens: words and punctuation marks)
    parent_tokens: int = 1024
    child_tokens: int = 256
    child_overlap_tokens: int = 32
    embed_batch_size: int = 64
    # retrieval
    dense_k: int = 30
    sparse_k: int = 30
    rrf_k: int = 60
    rerank_candidates: int = 20  # fused child chunks sent to the reranker
    top_k: int = 4  # parent chunks handed to the LLM
    # verification / decision gate
    verification_threshold: float = 0.95
    max_attempts: int = 2  # generation attempts before the deterministic fallback
    # caching
    cache_ttl_seconds: int = 3600

    def __post_init__(self):
        # The decision gate: a threshold of 0 (or NaN, or >1) would silently disable -- or make unreachable -- the
        # verification the whole product rests on, so refuse it instead of "working".
        if not (0.0 < self.verification_threshold <= 1.0):
            raise ValueError(f"verification_threshold must be in (0, 1], got {self.verification_threshold!r}")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.child_tokens > self.parent_tokens or self.child_overlap_tokens >= self.child_tokens:
            raise ValueError("require child_overlap_tokens < child_tokens <= parent_tokens")

    @classmethod
    def from_env(cls, prefix: str = "CONTEXTRAG_") -> "Settings":
        out = {}
        for f in fields(cls):
            raw = os.getenv(prefix + f.name.upper())
            if raw is not None:
                out[f.name] = type(f.default)(raw)
        return cls(**out)
