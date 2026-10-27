"""Retrieve -> generate -> verify with a self-correction loop and a deterministic fallback (decision gate)."""
from __future__ import annotations

import hashlib
import json
from typing import Any
import re
import logging
import threading
import time
from collections import defaultdict

from .audit import AuditLog, record_of
from .core.cache import Cache
from .core.config import Settings
from .core.ingestion import Ingestor
from .context.budget import ContextBudget, assemble_evidence, span_context
from .core.tracing import StageMetrics, Tracer, short_hash, tracer_from_env
from .core.models import Answer, ClaimVerdict, Context, Status
from .core.text import content_tokens, stem
from .generation.llm import LLM, NO_ANSWER, GenerationError
from .memory.store import ConversationStore, QueueFull
from .retrieval.hybrid_search import HybridIndex, HybridRetriever
from .retrieval.reranker import Reranker
from .verification.aliases import AliasDecl, Resolution, extract_alias_declarations, resolve
from .verification.conflicts import ConflictDetector, ambiguous_subjects, build_detector
from .verification.entities import absent_terms, named_phrases, named_terms
from .verification.injection import injection_markers
from .verification.safety import has_active_content
from .verification.nli_verifier import NLIModel, NLIVerifier, fallback_message


log = logging.getLogger("contextrag")


class ConversationUnavailable(RuntimeError):
    """A request named a conversation but no conversation memory is configured."""


class ContextRAG:
    def __init__(self, *, index: HybridIndex, embedder, reranker: Reranker, llm: LLM, nli: NLIModel,
                 audit: AuditLog | None = None, cache: Cache | None = None, settings: Settings | None = None,
                 memory: ConversationStore | None = None, compactor=None, conflicts: ConflictDetector | None = None,
                 tracer: Tracer | None = None):
        self.cfg = settings or Settings()
        self.index, self.embedder, self.llm, self.audit, self.cache = index, embedder, llm, audit, cache
        self.memory, self.compactor = memory, compactor
        self.reranker = reranker
        self.ingestor = Ingestor(index, embedder, self.cfg, cache)
        self.retriever = HybridRetriever(index, embedder, reranker, self.cfg)
        self.verifier = NLIVerifier(nli, self.cfg.verification_threshold)
        # an explicit detector wins; otherwise the configured one (a misconfigured "nli" mode fails here, not silently)
        self.conflict_detector = (conflicts if conflicts is not None
                                  else build_detector(self.cfg.conflict_detection, nli, self.cfg.conflict_threshold))
        self.budget = ContextBudget(self.cfg.context_budget_tokens, self.cfg.context_safety_margin)
        self._stats: dict[str, float] = defaultdict(float)
        self._stats_lock = threading.Lock()
        # Per-stage counters are always on; spans are exported only if a tracer is passed or CONTEXTRAG_TRACING selects one.
        self.tracer: Tracer = StageMetrics(self.count, tracer if tracer is not None else tracer_from_env())
        self.ingestor.tracer = self.retriever.tracer = self.verifier.tracer = self.tracer

    # -- public API -----------------------------------------------------------------------
    def ask(self, question: str, *, tenant: str, groups: frozenset[str] = frozenset(), user: str | None = None,
            top_k: int | None = None, conversation_id: str | None = None) -> Answer:
        with self.tracer.span("ask", tenant=short_hash(tenant)):  # the tenant appears in traces only as a short hash
            return self._ask(question, tenant, groups, user, top_k, conversation_id)

    def _ask(self, question: str, tenant: str, groups: frozenset[str], user: str | None, top_k: int | None,
             conversation_id: str | None) -> Answer:
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
            self.audit.write(record_of(tenant, user, answer, question_mode=self.cfg.audit_question,
                                       answer_mode=self.cfg.audit_answer), self.embedder.embed([question])[0])
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
        """Prometheus text exposition: the in-process counters, then the compaction worker's registry if one runs."""
        with self._stats_lock:
            items = sorted(self._stats.items())
        text = "".join(f"contextrag_{k} {v:g}\n" for k, v in items)
        errors = getattr(self.cache, "errors", None)
        if errors is not None:  # only a resilient (external) cache counts failures
            text += f"contextrag_cache_errors_total {errors:g}\n"
        if self.compactor is not None:
            from prometheus_client import generate_latest

            text += generate_latest(self.compactor.metrics.registry).decode()
        return text

    def erase_tenant(self, tenant: str, *, include_audit: bool = True) -> dict[str, int]:
        """Right-to-erasure: remove ALL of a tenant's documents, vectors, BM25 entries, cached answers and (unless
        ``include_audit`` is false, e.g. a legal hold) audit records. API keys are managed separately."""
        if not tenant:
            raise ValueError("tenant required")
        docs = self.ingestor.delete_tenant(tenant)  # also bumps the cache epoch: cached answers become unreachable
        records = self.audit.delete_tenant(tenant) if (self.audit and include_audit) else 0
        turns = self.memory.delete_tenant(tenant) if self.memory is not None else 0  # conversations hold user text too
        log.warning("tenant erased tenant=%s documents=%d audit_records=%d conversation_turns=%d",
                    tenant, docs, records, turns)
        return {"documents": docs, "audit_records": records, "conversation_turns": turns}

    def delete_document(self, tenant: str, doc_id: str) -> dict[str, int]:
        """Remove a document everywhere this pipeline controls: index, BM25, cache (by epoch) and the conversation turns
        built from it. The audit keeps references and digests, not the excerpt (see ``audit.record_of``)."""
        self.ingestor.delete(tenant=tenant, doc_id=doc_id)
        forgotten = self.memory.forget_document(tenant, doc_id) if self.memory is not None else {"turns": 0}
        log.info("document deleted tenant=%s conversation_turns=%d", tenant, forgotten["turns"])
        return forgotten

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
                       self.cfg.context_dedupe_jaccard, self.cfg.context_placement, self.cfg.require_question_entities,
                       self.cfg.resolve_entity_aliases, self.cfg.quarantine_suspected_injection,
                       self.cfg.ambiguity_policy, getattr(self.conflict_detector, "name", None),  # the detector in use, not only the setting
                       type(self.verifier.nli).__name__, getattr(self.verifier.nli, "model_name", None),
                       type(self.reranker).__name__, getattr(self.reranker, "model_name", None),
                       type(self.llm).__name__,
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
        cited = [s.parent.doc_id for s in answer.sources]  # provenance, so a deleted document takes its excerpts with it
        self.memory.append_turn(tenant, conversation_id, "assistant", answer.text, cites=cited)
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
    def _prepare(self, question, tenant, groups, top_k) -> tuple[list[Context], str, list[Context]]:
        """Evidence for ``question``: retrieval, quarantine of suspected injection, span fallback and the budget.
        Returns ``(contexts, "", retrieved)``, or ``([], reason, [])`` when nothing usable remains. ``retrieved`` is every
        passage that survived access control and quarantine, *before* de-duplication: two passages that differ by one
        word ("was notified" / "was not notified") look redundant to the budget, but they are exactly what a conflict
        check has to compare."""
        detailed = self.retriever.retrieve_detailed(question, tenant, groups, top_k)
        if not detailed:
            return [], "no relevant context retrieved", []
        if self.cfg.quarantine_suspected_injection:
            clean = [(ctx, child) for ctx, child in detailed if not injection_markers(ctx.parent.text)]
            if len(clean) < len(detailed):
                self.count("quarantined_passages_total", len(detailed) - len(clean))
            if not clean:
                return [], "retrieved passages were withheld as suspected prompt injection; human review needed", []
            detailed = clean
        with self.tracer.span("context", passages=len(detailed)):
            spans = {ctx.parent.id: span_context(ctx, child) for ctx, child in detailed}
            contexts = assemble_evidence([ctx for ctx, _ in detailed], self.budget,
                                         dedupe_threshold=self.cfg.context_dedupe_jaccard,
                                         placement=self.cfg.context_placement, spans=spans)
        if not contexts:
            return [], "retrieved context does not fit the evidence budget", []
        return contexts, "", [ctx for ctx, _ in detailed]

    def _alias_declarations(self, term: str, tenant: str, groups) -> list[AliasDecl]:
        """Alias declarations found in the passages that mention ``term``. The search is the caller's own ACL-filtered
        retrieval, and passages that look like prompt injection are not read, so neither access control nor the
        quarantine can be bypassed by asking about a name."""
        seen: set[str] = set()
        found: list[AliasDecl] = []
        for child, _ in self.retriever.search_children(term, tenant, groups)[: self.cfg.alias_lookup_k]:
            if child.parent_id in seen:
                continue
            seen.add(child.parent_id)
            parent = self.retriever.index.store.parent(child.parent_id, tenant)
            if parent is None or injection_markers(parent.text):
                continue
            found.extend(extract_alias_declarations(parent.text))
        return found

    def _clarification(self, question, contexts, ambiguous: Resolution, meta: dict) -> Answer:
        """The documents give one name to several entities. Choosing one would be arbitrary, so ask which is meant."""
        self.count("entity_ambiguous_total")
        meta = {**meta, "clarification": {"term": ambiguous.term, "candidates": list(ambiguous.canonicals)}}
        ans = self._fallback(question, contexts, [], 0, "the question names an ambiguous entity; clarification needed")
        ans.text = (f"La pregunta menciona «{ambiguous.term}», que los documentos asocian a más de una entidad: "
                    f"{'; '.join(ambiguous.canonicals)}. Indique a cuál se refiere.")
        ans.meta = meta
        return ans

    def _ambiguous_in_evidence(self, terms, contexts) -> Resolution | None:
        """A named term the evidence *does* mention can still be ambiguous: the retrieved passages declare it as an
        alias of two entities. Only the evidence in hand is read, so an ambiguity no retrieved passage shows is missed."""
        declarations = [d for c in contexts for d in extract_alias_declarations(c.parent.text)]
        for term in terms:
            r = resolve(term, declarations)
            if r.outcome == "ambiguous":
                return r
        return None

    def _resolve_entities(self, question, missing, tenant, groups, top_k, contexts, known=()):
        """Names in the question that need the documents' help. ``missing`` terms are absent from the evidence: either
        the documents declare them as aliases of one entity (retrieve again with the entity's name), declare them for
        several entities (ask which), or say nothing (abstain). ``known`` are aliases already declared in the evidence
        (the term is present, but the facts about the entity may live in a passage that only uses its full name).
        Returns an ``Answer`` to give back, or ``(contexts, question, meta)`` to continue."""
        not_found = "the evidence does not mention a named term from the question"
        if not self.cfg.resolve_entity_aliases:
            self.count("entity_gate_rejections_total")  # a count only: the terms come from the user's question
            return self._fallback(question, contexts, [], 0, not_found)
        resolutions = list(known) + [resolve(t, self._alias_declarations(t, tenant, groups)) for t in missing]
        meta: dict = {"entity_resolution": [{"term": r.term, "outcome": r.outcome, "canonical": list(r.canonicals)}
                                            for r in resolutions]}
        ambiguous = next((r for r in resolutions if r.outcome == "ambiguous"), None)
        if ambiguous is not None:
            return self._clarification(question, contexts, ambiguous, meta)
        if any(r.outcome == "unknown" for r in resolutions):
            self.count("entity_gate_rejections_total")
            ans = self._fallback(question, contexts, [], 0, not_found)
            ans.meta = meta
            return ans
        canonical = [c for r in resolutions for c in r.canonicals]
        new_contexts, _, _ = self._prepare(f"{question} {' '.join(canonical)}", tenant, groups, top_k)
        if not new_contexts or absent_terms(canonical, (c.parent.text for c in new_contexts)):
            self.count("entity_gate_rejections_total")  # the alias exists, but nothing describes the entity itself
            ans = self._fallback(question, contexts, [], 0, "the documented entity is not described by the evidence")
            ans.meta = meta
            return ans
        self.count("entity_alias_resolved_total", len(resolutions))
        asked = question
        for r in resolutions:
            asked = re.sub(re.escape(r.term), r.canonicals[0], asked, flags=re.IGNORECASE)
        return new_contexts, asked, meta

    def _entity_step(self, question, tenant, groups, top_k, contexts, asked, meta):
        """Names in the question against the evidence: aliases, ambiguity, unknown entities. Returns
        ``(early_answer_or_None, contexts, asked, meta)``; an early answer ends the request."""
        named = named_terms(question)
        missing = absent_terms(named, (c.parent.text for c in contexts))
        known: list[Resolution] = []
        if self.cfg.resolve_entity_aliases:
            # a term the evidence mentions can still be an alias (the facts may use the full name) or ambiguous
            declarations = [d for c in contexts for d in extract_alias_declarations(c.parent.text)]
            present = [resolve(t, declarations) for t in named if t not in missing]
            ambiguous = next((r for r in present if r.outcome == "ambiguous"), None)
            if ambiguous is not None:
                return self._clarification(question, contexts, ambiguous, {}), contexts, asked, meta
            known = [r for r in present if r.outcome == "alias"]
        if missing or known:
            resolved = self._resolve_entities(question, missing, tenant, groups, top_k, contexts, known)
            if isinstance(resolved, Answer):
                return resolved, contexts, asked, meta
            contexts, asked, meta = resolved
        return None, contexts, asked, meta

    def _answer(self, question, tenant, groups, top_k, memory_text: str | None = None) -> Answer:
        contexts, why, retrieved = self._prepare(question, tenant, groups, top_k)
        if not contexts:
            return self._fallback(question, [], [], 0, why)
        meta: dict[str, Any] = {}
        asked = question  # ``asked`` is what the model sees; the answer keeps the user's original wording
        focus: list[str] = []
        if self.cfg.require_question_entities:
            with self.tracer.span("entities"):
                early, contexts, asked, meta = self._entity_step(question, tenant, groups, top_k, contexts, asked, meta)
            if early is not None:
                return early
            focus = named_phrases(asked)

        if self.conflict_detector is not None:
            with self.tracer.span("conflicts", passages=len(retrieved)):
                found = self.conflict_detector.detect(asked, retrieved)
            if found:
                return self._conflict_answer(question, contexts, found, meta)

        if self.cfg.ambiguity_policy != "off":
            subjects = ambiguous_subjects(asked, retrieved)
            if subjects:
                self.count("ambiguous_subject_total")
                if self.cfg.ambiguity_policy == "clarify":
                    ans = self._fallback(question, contexts, [], 0, "the question does not say which matter it refers to")
                    ans.text = ("La pregunta no indica a cuál de estos asuntos se refiere: "
                                f"{'; '.join(subjects)}. Indique cuál.")
                    ans.meta = {**meta, "clarification": {"kind": "ambiguous_subject", "candidates": subjects}}
                    return ans
                meta = {**meta, "scope_note": {"kind": "ambiguous_subject", "subjects": subjects,
                                               "note": "The question names no subject; the evidence covers several, so "
                                                       "the answer may refer to only some of them."}}

        feedback, attempts, blocked = None, 0, False
        verdicts: list[ClaimVerdict] = []
        for attempts in range(1, self.cfg.max_attempts + 1):
            try:
                extra = {"memory": memory_text} if memory_text else {}
                with self.tracer.span("generate", attempt=attempts):
                    text = self.llm.generate(asked, contexts, feedback, **extra).strip()
            except GenerationError:
                raise
            except Exception as e:
                # Only the class: the message of an arbitrary exception can hold paths, keys or document text.
                # The original stays chained as __cause__ for internal debugging, never for a client or a log line.
                raise GenerationError(type(e).__name__) from e
            if not text or text == NO_ANSWER:
                return self._fallback(question, contexts, [], attempts, "model reported insufficient context")
            if has_active_content(text):  # images, scripts, embedded HTML: refused whatever the evidence says
                self.count("active_content_rejections_total")
                blocked, verdicts = True, []
                feedback = "- the answer contained an image, script or embedded HTML; answer in plain text only"
                continue
            blocked = False
            with self.tracer.span("verify", attempt=attempts):
                verdicts = self._relevance_gate(asked, self.verifier.verify(text, contexts))
            verdicts = self._focus_gate(focus, verdicts, contexts)
            if verdicts and all(v.supported for v in verdicts):
                return Answer(Status.VERIFIED, question, text, min(v.score for v in verdicts), verdicts,
                              contexts, attempts, meta=meta)
            feedback = ("\n".join(f"- {v.claim} ({v.reason})" for v in verdicts if not v.supported)
                        or "- the answer contained no verifiable claims")
        if blocked:
            return self._fallback(question, contexts, [], attempts, "the answer contained active content")
        return self._fallback(question, contexts, verdicts, attempts, "verification below threshold",
                              min((v.score for v in verdicts), default=0.0))

    def _conflict_answer(self, question, contexts, found, meta: dict) -> Answer:
        """Sources disagree. Choosing one would present a guess as a fact, so nothing is chosen: the answer says so, and
        the sources (document, page, offsets) are in ``meta`` for a person to compare."""
        self.count("conflicts_detected_total")
        ans = self._fallback(question, contexts, [], 0, "conflicting sources: human review needed")
        ans.text = ("Los documentos recuperados contienen información que parece contradictoria sobre esta pregunta. "
                    "No se elige ninguna versión; las fuentes implicadas se indican en el detalle y requieren revisión "
                    "humana.")
        ans.meta = {**meta, "conflicts": [c.to_dict() for c in found], "needs_human_review": True}
        return ans

    def _focus_gate(self, focus: list[str], verdicts, contexts):
        """Faithful is not enough, and neither is relevant: when the question names an entity, each claim must rest on a
        passage that mentions it. A true statement about another entity's record is not part of the answer."""
        if not focus:
            return verdicts
        for v in verdicts:
            if not v.supported or v.evidence is None:
                continue
            e = v.evidence
            ctx = next((c for c in contexts if c.parent.doc_id == e.doc_id and c.parent.start <= e.start
                        and e.end <= c.parent.end), None)
            if ctx is not None and len(absent_terms(focus, [ctx.parent.text])) == len(focus):
                self.count("entity_consistency_rejections_total")
                v.supported, v.evidence, v.score = False, None, 0.0
                v.reason = "evidence is not about the entity named in the question"
        return verdicts

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
