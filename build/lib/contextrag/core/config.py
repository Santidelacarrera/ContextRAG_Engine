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

    @classmethod
    def from_env(cls, prefix: str = "CONTEXTRAG_") -> "Settings":
        out = {}
        for f in fields(cls):
            raw = os.getenv(prefix + f.name.upper())
            if raw is not None:
                out[f.name] = type(f.default)(raw)
        return cls(**out)
