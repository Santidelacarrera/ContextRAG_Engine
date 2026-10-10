"""Retrieve -> generate -> verify with a self-correction loop and a deterministic fallback (decision gate)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import Counter

from .audit import AuditLog, record_of
from .core.cache import Cache
from .core.config import Settings
from .core.ingestion import Ingestor
from .core.models import Answer, Status
from .generation.llm import LLM, NO_ANSWER
from .retrieval.hybrid_search import HybridIndex, HybridRetriever
from .retrieval.reranker import Reranker
from .verification.nli_verifier import NLIModel, NLIVerifier, fallback_message


log = logging.getLogger("contextrag")


class ContextRAG:
    def __init__(self, *, index: HybridIndex, embedder, reranker: Reranker, llm: LLM, nli: NLIModel,
                 audit: AuditLog | None = None, cache: Cache | None = None, settings: Settings | None = None):
        self.cfg = settings or Settings()
        self.index, self.embedder, self.llm, self.audit, self.cache = index, embedder, llm, audit, cache
        self.reranker = reranker
        self.ingestor = Ingestor(index, embedder, self.cfg, cache)
        self.retriever = HybridRetriever(index, embedder, reranker, self.cfg)
        self.verifier = NLIVerifier(nli, self.cfg.verification_threshold)
        self._stats: Counter = Counter()
        self._stats_lock = threading.Lock()

    # -- public API -----------------------------------------------------------------------
    def ask(self, question: str, *, tenant: str, groups: frozenset[str] = frozenset(), user: str | None = None,
            top_k: int | None = None) -> Answer:
        t0 = time.perf_counter()
        answer = self._cached(question, tenant, groups, top_k)
        if answer is None:
            answer = self._answer(question, tenant, groups, top_k)
            self._store(question, tenant, groups, top_k, answer)
        if self.audit:
            self.audit.write(record_of(tenant, user, answer), self.embedder.embed([question])[0])
        ms = (time.perf_counter() - t0) * 1000
        with self._stats_lock:
            self._stats[f'queries_total{{status="{answer.status.value}"}}'] += 1
            self._stats["query_cache_hits_total"] += answer.cached
            self._stats["query_latency_ms_sum"] += ms
            self._stats["query_latency_ms_count"] += 1
        log.info("query tenant=%s status=%s confidence=%.2f attempts=%d cached=%s sources=%d ms=%.0f reason=%r",
                 tenant, answer.status.value, answer.confidence, answer.attempts, answer.cached,
                 len(answer.sources), ms, answer.reason)
        return answer

    def metrics_text(self) -> str:
        """Prometheus text exposition of in-process counters."""
        with self._stats_lock:
            items = sorted(self._stats.items())
        return "".join(f"contextrag_{k} {v:g}\n" for k, v in items)

    def health(self) -> dict[str, bool]:
        parts = {"index": self.index.health()}
        if self.cache:
            parts["cache"] = self.cache.health()
        if self.audit:
            parts["audit"] = self.audit.health()
        return parts

    # -- caching (keys include a per-tenant epoch bumped on every ingest) -------------------
    def _key(self, question, tenant, groups, top_k) -> str | None:
        if not self.cache:
            return None
        epoch = self.cache.get(f"epoch:{tenant}") or "0"
        raw = json.dumps([tenant, epoch, sorted(groups), question, top_k, self.cfg.verification_threshold])
        return "ans:" + hashlib.sha256(raw.encode()).hexdigest()

    def _cached(self, question, tenant, groups, top_k) -> Answer | None:
        key = self._key(question, tenant, groups, top_k)
        hit = self.cache.get(key) if key else None
        if hit is None:
            return None
        a = Answer.from_dict(json.loads(hit))
        a.cached = True
        return a

    def _store(self, question, tenant, groups, top_k, answer: Answer) -> None:
        key = self._key(question, tenant, groups, top_k)
        if key and answer.status is Status.VERIFIED:  # never pin a transient fallback in the cache
            self.cache.set(key, json.dumps(answer.to_dict()), self.cfg.cache_ttl_seconds)

    # -- core loop --------------------------------------------------------------------------
    def _answer(self, question, tenant, groups, top_k) -> Answer:
        contexts = self.retriever.retrieve(question, tenant, groups, top_k)
        if not contexts:
            return self._fallback(question, [], [], 0, "no relevant context retrieved")

        feedback, attempts, verdicts = None, 0, []
        for attempts in range(1, self.cfg.max_attempts + 1):
            text = self.llm.generate(question, contexts, feedback).strip()
            if not text or text == NO_ANSWER:
                return self._fallback(question, contexts, [], attempts, "model reported insufficient context")
            verdicts = self._relevance_gate(question, self.verifier.verify(text, contexts))
            if verdicts and all(v.supported for v in verdicts):
                return Answer(Status.VERIFIED, question, text, min(v.score for v in verdicts), verdicts,
                              contexts, attempts)
            feedback = ("\n".join(f"- {v.claim} ({v.reason})" for v in verdicts if not v.supported)
                        or "- the answer contained no verifiable claims")
        return self._fallback(question, contexts, verdicts, attempts, "verification below threshold",
                              min((v.score for v in verdicts), default=0.0))

    def _relevance_gate(self, question: str, verdicts):
        """Faithful is not enough: the span backing each claim must also be relevant to the question.
        This stops an answer that merely parrots a poisoned/off-topic sentence of a retrieved document
        (e.g. "ignore previous instructions and say X"). Applied when the reranker declares ``min_score``."""
        floor = getattr(self.reranker, "min_score", None)
        checked = [v for v in verdicts if v.supported and v.evidence]
        if floor is None or not checked:
            return verdicts
        for v, s in zip(checked, self.reranker.score(question, [v.evidence.text for v in checked])):
            if s <= floor:
                v.supported, v.evidence, v.reason = False, None, "evidence not relevant to the question"
        return verdicts

    def _fallback(self, question, contexts, verdicts, attempts, reason, confidence=0.0) -> Answer:
        return Answer(Status.FALLBACK, question, fallback_message(self.cfg.verification_threshold), confidence,
                      verdicts, contexts, attempts, reason)
