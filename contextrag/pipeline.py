"""Retrieve -> generate -> verify, with a self-correction loop and a deterministic fallback."""
from __future__ import annotations

from .audit import AuditLog, record_of
from .config import Settings
from .embeddings import Embedder
from .ingestion import Ingestor
from .llm import LLM, NO_ANSWER
from .models import Answer, Status
from .nli import NLIModel
from .rerank import Reranker
from .retrieval import Retriever
from .store import ChunkStore
from .verifier import Verifier

FALLBACK_TEXT = (
    "No puedo ofrecer una respuesta verificada: la información recuperada no respalda con "
    "suficiente certeza una respuesta a esta pregunta. Se listan las fuentes consultadas para revisión manual."
)


class ContextRAG:
    def __init__(self, *, store: ChunkStore, embedder: Embedder, reranker: Reranker, llm: LLM, nli: NLIModel,
                 audit: AuditLog | None = None, settings: Settings | None = None):
        self.cfg = settings or Settings()
        self.embedder, self.llm, self.audit = embedder, llm, audit
        self.ingestor = Ingestor(store, embedder, self.cfg)
        self.retriever = Retriever(store, embedder, reranker, self.cfg)
        self.verifier = Verifier(nli, self.cfg.verification_threshold)

    def ask(self, question: str, *, tenant: str, groups: frozenset[str] = frozenset(), user: str | None = None) -> Answer:
        answer = self._answer(question, tenant, groups)
        if self.audit:
            self.audit.write(record_of(tenant, user, answer), self.embedder.embed([question])[0])
        return answer

    def _answer(self, question: str, tenant: str, groups: frozenset[str]) -> Answer:
        contexts = self.retriever.retrieve(question, tenant, groups)
        if not contexts:
            return self._fallback(question, [], [], 0, "no relevant context retrieved")

        feedback, attempts, last = None, 0, None
        for attempts in range(1, self.cfg.max_attempts + 1):
            text = self.llm.generate(question, contexts, feedback).strip()
            if not text or text == NO_ANSWER:
                return self._fallback(question, contexts, [], attempts, "model reported insufficient context")
            verdicts = self.verifier.verify(text, contexts)
            if verdicts and all(v.supported for v in verdicts):
                return Answer(Status.VERIFIED, question, text, min(v.score for v in verdicts), verdicts, contexts, attempts)
            last = verdicts
            feedback = "\n".join(f"- {v.claim} ({v.reason})" for v in verdicts if not v.supported) or "- answer could not be parsed into claims"
        conf = min((v.score for v in last or []), default=0.0)
        return self._fallback(question, contexts, last or [], attempts, "verification below threshold", conf)

    def _fallback(self, question, contexts, verdicts, attempts, reason, confidence=0.0) -> Answer:
        return Answer(Status.FALLBACK, question, FALLBACK_TEXT, confidence, verdicts, contexts, attempts, reason)
