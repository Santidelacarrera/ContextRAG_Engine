from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import MISSING, dataclass, fields
from typing import Any


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
    # evidence budget (tokens, same unit as the chunker): the passages handed to the LLM never exceed it
    context_budget_tokens: int = 4096  # == top_k * parent_tokens at the defaults: no behaviour change
    context_safety_margin: float = 0.10  # fraction of the budget held back for tokenizer estimation error
    context_dedupe_jaccard: float = 0.8  # parents this similar to a better-ranked one are dropped
    # descending keeps the pre-existing order; edges is opt-in until an LLM-based experiment shows a gain
    context_placement: str = "descending"
    # reranking
    reranker_timeout_seconds: float = 10.0  # a reranker call that exceeds this fails the request (never guessed)
    # conversational memory
    memory_compact_threshold_tokens: int = 2000  # uncompacted history above this schedules a compaction
    memory_keep_recent_turns: int = 6  # the most recent turns are never compacted
    memory_workers: int = 2  # bounded summarization concurrency
    memory_queue_limit: int = 64  # pending compaction jobs; beyond this, scheduling is refused
    memory_job_max_attempts: int = 3
    memory_lease_seconds: int = 300  # a running job not finished within this is considered abandoned
    memory_history_tokens: int = 1000  # recent turns included in the prompt, newest first, within this budget
    memory_relevant_k: int = 5  # memories included per request: the ones that match the question

    def __post_init__(self):
        # The decision gate: a threshold of 0 (or NaN, or >1) would silently disable -- or make unreachable -- the
        # verification the whole product rests on, so refuse it instead of "working".
        if not (0.0 < self.verification_threshold <= 1.0):
            raise ValueError(f"verification_threshold must be in (0, 1], got {self.verification_threshold!r}")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.child_tokens > self.parent_tokens or self.child_overlap_tokens >= self.child_tokens:
            raise ValueError("require child_overlap_tokens < child_tokens <= parent_tokens")
        if self.context_budget_tokens < 1:
            raise ValueError("context_budget_tokens must be >= 1")
        if not (0.0 <= self.context_safety_margin < 1.0):
            raise ValueError("context_safety_margin must be in [0, 1)")
        if not (0.0 < self.context_dedupe_jaccard <= 1.0):
            raise ValueError("context_dedupe_jaccard must be in (0, 1]")
        if self.context_placement not in ("edges", "descending"):
            raise ValueError("context_placement must be 'edges' or 'descending'")
        if self.reranker_timeout_seconds <= 0:
            raise ValueError("reranker_timeout_seconds must be > 0")
        if self.memory_keep_recent_turns < 0 or self.memory_compact_threshold_tokens < 1:
            raise ValueError("memory thresholds must be positive (keep_recent_turns may be 0)")
        if self.memory_workers < 1 or self.memory_queue_limit < 1 or self.memory_job_max_attempts < 1:
            raise ValueError("memory worker, queue and attempt limits must be >= 1")
        if self.memory_history_tokens < 0 or self.memory_relevant_k < 0:
            raise ValueError("memory_history_tokens and memory_relevant_k must be >= 0")
        if self.memory_lease_seconds <= 0:
            raise ValueError("memory_lease_seconds must be > 0")

    @classmethod
    def from_env(cls, prefix: str = "CONTEXTRAG_") -> "Settings":
        out: dict[str, Any] = {}
        for f in fields(cls):
            raw = os.getenv(prefix + f.name.upper())
            if raw is not None:
                default = f.default if f.default is not MISSING else None
                convert: Callable[[str], Any] = type(default) if default is not None else str
                out[f.name] = convert(raw)
        return cls(**out)
