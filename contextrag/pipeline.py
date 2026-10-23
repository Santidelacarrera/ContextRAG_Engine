"""Retrieve -> generate -> verify with a self-correction loop and a deterministic fallback (decision gate)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict

from .audit import AuditLog, record_of
from .core.cache import Cache
from .core.config import Settings
from .core.ingestion import Ingestor
from .context.budget import ContextBudget, assemble_evidence, span_context
from .core.models import Answer, Status
from .core.text import content_tokens, stem
from .generation.llm import LLM, NO_ANSWER, GenerationError
from .memory.store import ConversationStore, QueueFull
from .retrieval.hybrid_search import HybridIndex, HybridRetriever
from .retrieval.reranker import Reranker
from .verification.nli_verifier import NLIModel, NLIVerifier, fallback_message


log = logging.getLogger("contextrag")


class ConversationUnavailable(RuntimeError):
    """A request named a conversation but no conversation memory is configured."""


class ContextRAG:
    def __init__(self, *, index: HybridIndex, embedder, reranker: Reranker, llm: LLM, nli: NLIModel,
                 audit: AuditLog | None = None, cache: Cache | None = None, settings: Settings | None = None,
                 memory: ConversationStore | None = None, compactor=None):
        self.cfg = settings or Settings()
        self.index, self.embedder, self.llm, self.audit, self.cache = index, embedder, llm, audit, cache
        self.memory, self.compactor = memory, compactor
        self.reranker = reranker
        self.ingestor = Ingestor(index, embedder, self.cfg, cache)
        self.retriever = HybridRetriever(index, embedder, reranker, self.cfg)
        self.verifier = NLIVerifier(nli, self.cfg.verification_threshold)
        self.budget = ContextBudget(self.cfg.context_budget_tokens, self.cfg.context_safety_margin)
        self._stats: dict[str, float] = defaultdict(float)
        self._stats_lock = threading.Lock()

    # -- public API -----------------------------------------------------------------------
    def ask(self, question: str, *, tenant: str, groups: frozenset[str] = frozenset(), user: str | None = None,
            top_k: int | None = None, conversation_id: str | None = None) -> Answer:
        t0 = time.perf_counter()
        if conversation_id is None:
            answer = self._cached(question, tenant, groups, top_k)
            if answer is None:
                answer = self._answer(question, tenant, groups, top_k)
                self._store(question, tenant, groups, top_k, answer)
        else:
            # Never cached: the answer depends on the conversation's history, not only on the question.
            answer = self._conversational(question, tenant, groups, top_k, conversation_id)
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

    def close(self) -> None:
        """Stop background compaction and close the conversation store (call on application shutdown)."""
        if self.compactor is not None:
            self.compactor.close()
        if self.memory is not None:
            self.memory.close()

    def count(self, key: str, n: float = 1) -> None:
        """Increment an in-process counter exposed by ``metrics_text`` (e.g. ``retrieve_rejected_total{reason="x"}``)."""
        with self._stats_lock:
            self._stats[key] += n

    def metrics_text(self) -> str:
        """Prometheus text exposition of in-process counters."""
        with self._stats_lock:
            items = sorted(self._stats.items())
        return "".join(f"contextrag_{k} {v:g}\n" for k, v in items)

    def erase_tenant(self, tenant: str, *, include_audit: bool = True) -> dict[str, int]:
        """Right-to-erasure: remove ALL of a tenant's documents, vectors, BM25 entries, cached answers and (unless
        ``include_audit`` is false, e.g. a legal hold) audit records. API keys are managed separately."""
        if not tenant:
            raise ValueError("tenant required")
        docs = self.ingestor.delete_tenant(tenant)  # also bumps the cache epoch: cached answers become unreachable
        records = self.audit.delete_tenant(tenant) if (self.audit and include_audit) else 0
        log.warning("tenant erased tenant=%s documents=%d audit_records=%d", tenant, docs, records)
        return {"documents": docs, "audit_records": records}

    def purge_audit(self, older_than_days: float) -> int:
        """Audit retention: delete records older than ``older_than_days`` (all tenants)."""
        if older_than_days < 0:
            raise ValueError("older_than_days must be >= 0")
        return self.audit.purge(older_than_days * 86400) if self.audit else 0

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
        # Everything that changes the answer is part of the key: a cached answer built with another budget,
        # placement, verifier threshold or generator must not be served as if it were current.
        fingerprint = [self.cfg.verification_threshold, self.cfg.context_budget_tokens, self.cfg.context_safety_margin,
                       self.cfg.context_dedupe_jaccard, self.cfg.context_placement, type(self.llm).__name__,
                       getattr(self.llm, "model", None)]
        raw = json.dumps([tenant, epoch, sorted(groups), question, top_k, fingerprint])
        return "ans:" + hashlib.sha256(raw.encode()).hexdigest()

    def _cached(self, question, tenant, groups, top_k) -> Answer | None:
        key = self._key(question, tenant, groups, top_k)
        hit = self.cache.get(key) if (key and self.cache) else None
        if hit is None:
            return None
        a = Answer.from_dict(json.loads(hit))
        a.cached = True
        return a

    def _store(self, question, tenant, groups, top_k, answer: Answer) -> None:
        key = self._key(question, tenant, groups, top_k)
        if key and self.cache and answer.status is Status.VERIFIED:  # never pin a transient fallback in the cache
            self.cache.set(key, json.dumps(answer.to_dict()), self.cfg.cache_ttl_seconds)

    # -- conversations ------------------------------------------------------------------------
    def _conversational(self, question, tenant, groups, top_k, conversation_id: str) -> Answer:
        if self.memory is None:
            raise ConversationUnavailable("conversation memory is not configured (MEMORY_DB)")
        self.memory.create_conversation(tenant, conversation_id)  # idempotent
        memory_text = self._memory_context(question, tenant, conversation_id)
        answer = self._answer(question, tenant, groups, top_k, memory_text=memory_text)
        # Both turns are committed after the answer exists. Failed or fallback answers are stored too: the user
        # did ask, and the next turn should know what was said.
        self.memory.append_turn(tenant, conversation_id, "user", question)
        self.memory.append_turn(tenant, conversation_id, "assistant", answer.text)
        if self.compactor is not None:
            try:
                self.compactor.schedule(tenant, conversation_id, threshold_tokens=self.cfg.memory_compact_threshold_tokens,
                                        keep_recent_turns=self.cfg.memory_keep_recent_turns)
            except QueueFull:
                log.warning("compaction deferred: queue full tenant=%s", tenant)  # never fails the request
        return answer

    def _memory_context(self, question: str, tenant: str, conversation_id: str) -> str | None:
        def overlap(query: str, text: str) -> float:  # the same term-coverage measure as the lexical reranker
            q = {stem(t) for t in content_tokens(query)}
            return float(len(q & {stem(t) for t in content_tokens(text)}))

        memory = self.memory
        if memory is None:  # guarded by _conversational; keeps the type narrow for readers and mypy
            raise ConversationUnavailable("conversation memory is not configured (MEMORY_DB)")
        memories = memory.relevant_memories(tenant, conversation_id, question, self.cfg.memory_relevant_k, overlap)
        window = memory.history_window(tenant, conversation_id, self.cfg.memory_history_tokens)
        lines = [f"- ({m.memory_type}, user-stated) {m.content}" for m in memories]
        if window:
            lines.append("Recent turns:")
            lines.extend(f"{t.role}: {t.content}" for t in window)
        return "\n".join(lines) or None

    # -- core loop --------------------------------------------------------------------------
    def _answer(self, question, tenant, groups, top_k, memory_text: str | None = None) -> Answer:
        detailed = self.retriever.retrieve_detailed(question, tenant, groups, top_k)
        if not detailed:
            return self._fallback(question, [], [], 0, "no relevant context retrieved")
        spans = {ctx.parent.id: span_context(ctx, child) for ctx, child in detailed}
        contexts = assemble_evidence([ctx for ctx, _ in detailed], self.budget,
                                     dedupe_threshold=self.cfg.context_dedupe_jaccard,
                                     placement=self.cfg.context_placement, spans=spans)
        if not contexts:
            return self._fallback(question, [], [], 0, "retrieved context does not fit the evidence budget")

        feedback, attempts, verdicts = None, 0, []
        for attempts in range(1, self.cfg.max_attempts + 1):
            try:
                extra = {"memory": memory_text} if memory_text else {}
                text = self.llm.generate(question, contexts, feedback, **extra).strip()
            except GenerationError:
                raise
            except Exception as e:
                raise GenerationError(f"{type(e).__name__}: {e}") from e
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
        for v, s in zip(checked, self.reranker.score(question, [v.evidence.text for v in checked]), strict=True):
            if s <= floor:
                v.supported, v.evidence, v.score = False, None, 0.0
                v.reason = "evidence not relevant to the question"
        return verdicts

    def _fallback(self, question, contexts, verdicts, attempts, reason, confidence=0.0) -> Answer:
        return Answer(Status.FALLBACK, question, fallback_message(self.cfg.verification_threshold), confidence,
                      verdicts, contexts, attempts, reason)
