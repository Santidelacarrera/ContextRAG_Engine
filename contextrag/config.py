from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    # chunking
    parent_size: int = 1800
    child_size: int = 400
    child_overlap: int = 80
    # retrieval
    dense_k: int = 30
    sparse_k: int = 30
    rerank_top_n: int = 8
    max_contexts: int = 4
    rrf_k: int = 60
    # verification
    verification_threshold: float = 0.95
    max_attempts: int = 2  # generation attempts before falling back
