"""Component wiring. ``offline()`` needs no models or services; ``from_env()`` picks
backends from environment variables (used by the API server and docker-compose)."""
from __future__ import annotations

import os

from .audit import AuditLog, InMemoryAudit
from .core.cache import Cache, CachedEmbedder, InMemoryCache
from .core.config import Settings
from .core.embeddings import Embedder, HashingEmbedder
from .generation.llm import LLM, ExtractiveLLM
from .pipeline import ContextRAG
from .retrieval.bm25 import BM25Index, IncrementalBM25Index
from .retrieval.hybrid_search import HybridIndex
from .retrieval.reranker import LexicalReranker, Reranker
from .retrieval.vector_store import InMemoryVectorStore, VectorStore
from .verification.nli_verifier import LexicalEntailment, NLIModel


def _opt_float(v: str | None) -> float | None:
    return float(v) if v not in (None, "") else None


def offline(settings: Settings | None = None, llm=None, *, tracer=None) -> ContextRAG:
    """Self-contained stack: hashing embeddings, in-memory stores, lexical rerank/NLI, extractive LLM."""
    return ContextRAG(index=HybridIndex(InMemoryVectorStore()), embedder=HashingEmbedder(),
                      reranker=LexicalReranker(), llm=llm or ExtractiveLLM(), nli=LexicalEntailment(),
                      audit=InMemoryAudit(), cache=InMemoryCache(), settings=settings, tracer=tracer)


def from_env() -> ContextRAG:
    """Backends are chosen by env vars:

    EMBEDDER=hashing|bge (EMBED_MODEL)   RERANKER=lexical|cross-encoder|cohere (COHERE_API_KEY, RERANK_MIN_SCORE: relevance floor)
    NLI=lexical|deberta (NLI_MODEL)                  LLM=extractive|anthropic|ollama (LLM_MODEL, OLLAMA_HOST)
    QDRANT_URL (else embedded in-memory)  MEMORY_DB (SQLite path; conversations off without it) POSTGRES_DSN (else in-memory audit; POSTGRES_ADMIN_DSN = owner for schema/retention)  REDIS_URL (else in-memory cache)
    BM25=rank|incremental (rank = rank_bm25, rebuilt after changes; incremental = same scores, fast under frequent ingests)
    MASTER_KEY (enables per-tenant encryption of chunk text at rest)  CONTEXTRAG_* (see Settings)
    """
    env = os.getenv
    cfg = Settings.from_env()

    from .core.security import TenantCipher

    from .core.security import load_master_keys

    try:
        master_key, previous = load_master_keys()  # MASTER_KEY or MASTER_KEY_FILE; refuses unsafe file permissions
    except (ValueError, OSError) as exc:  # the startup contract is RuntimeError; the message names no key
        raise RuntimeError(str(exc)) from None
    cipher = TenantCipher(master_key, previous) if master_key else None

    cache: Cache = InMemoryCache()
    redis_url = env("REDIS_URL")
    if redis_url:
        from .core.cache import EncryptedCache, RedisCache, ResilientCache

        cache = RedisCache(redis_url)
        if cipher is not None:  # answers and evidence must not sit in Redis in the clear when a key is configured
            cache = EncryptedCache(cache, cipher)
        cache = ResilientCache(cache)  # outermost: a Redis outage degrades to "no cache" instead of failing requests

    embedder: Embedder
    if env("EMBEDDER", "hashing") == "bge":
        from .core.embeddings import SentenceTransformerEmbedder

        name = env("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
        embedder = CachedEmbedder(SentenceTransformerEmbedder(name), cache, name)
    else:
        embedder = HashingEmbedder()

    store: VectorStore
    if env("QDRANT_URL") or cipher:
        from qdrant_client import QdrantClient

        from .retrieval.vector_store import QdrantVectorStore

        client = QdrantClient(url=env("QDRANT_URL"), api_key=env("QDRANT_API_KEY") or None) if env("QDRANT_URL") else None
        store = QdrantVectorStore(embedder.dim, client=client, cipher=cipher)
    else:
        store = InMemoryVectorStore()

    rer = env("RERANKER", "lexical")
    reranker: Reranker
    if rer == "cohere":
        from .retrieval.reranker import CohereReranker

        reranker = CohereReranker(env("COHERE_API_KEY"), min_score=_opt_float(env("RERANK_MIN_SCORE")))
    elif rer == "cross-encoder":
        from .retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(env("RERANK_MODEL", "BAAI/bge-reranker-large"),
                                        min_score=_opt_float(env("RERANK_MIN_SCORE")))
    else:
        reranker = LexicalReranker()
    if rer != "lexical":  # remote or model-backed rerankers get a hard timeout; the offline one is in-process
        from .retrieval.reranker import BoundedReranker

        reranker = BoundedReranker(reranker, cfg.reranker_timeout_seconds)

    nli: NLIModel
    if env("NLI", "lexical") == "deberta":
        from .verification.nli_verifier import DebertaNLI

        nli = DebertaNLI(env("NLI_MODEL") or "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                         revision=env("NLI_MODEL_REVISION") or None)
    else:
        nli = LexicalEntailment()

    llm_kind = env("LLM", "extractive")
    llm: LLM
    if llm_kind == "anthropic":
        from .generation.llm import AnthropicLLM

        llm = AnthropicLLM(env("LLM_MODEL"))
    elif llm_kind == "ollama":
        from .generation.llm import OllamaLLM

        llm = OllamaLLM(env("LLM_MODEL", "llama3"), env("OLLAMA_HOST", "http://localhost:11434"))
    else:
        llm = ExtractiveLLM()

    audit: AuditLog
    dsn = env("POSTGRES_DSN")
    if dsn:
        from .audit import PostgresAudit

        audit = PostgresAudit(dsn, embedder.dim, admin_dsn=env("POSTGRES_ADMIN_DSN") or None)
    else:
        audit = InMemoryAudit()

    bm25: BM25Index | IncrementalBM25Index | None = None
    if env("BM25", "rank") == "incremental":
        from .retrieval.bm25 import IncrementalBM25Index

        bm25 = IncrementalBM25Index(loader=lambda tenant: store.iter_children(tenant))
    memory = compactor = None
    memory_path = env("MEMORY_DB")
    if memory_path:  # durable conversations: a SQLite file path, e.g. /data/memory.db
        from .memory.store import ConversationStore
        from .memory.summarizer import ExtractiveSummarizer
        from .memory.worker import CompactionWorker

        plaintext = env("MEMORY_PLAINTEXT") == "1"  # explicit opt-out, for development only
        if cipher is None and not plaintext:
            raise RuntimeError("MEMORY_DB is set but MASTER_KEY is not: conversations are not stored in plaintext. "
                               "Set MASTER_KEY, or MEMORY_PLAINTEXT=1 for development.")
        memory = ConversationStore(memory_path, cipher=cipher, plaintext=plaintext,
                                   max_attempts=cfg.memory_job_max_attempts, queue_limit=cfg.memory_queue_limit)
        compactor = CompactionWorker(memory, ExtractiveSummarizer(), workers=cfg.memory_workers,
                                     lease_seconds=cfg.memory_lease_seconds)
        compactor.recover()  # jobs left running by a previous crash resume here
    return ContextRAG(index=HybridIndex(store, bm25), embedder=embedder, reranker=reranker, llm=llm, nli=nli,
                      audit=audit, cache=cache, settings=cfg, memory=memory, compactor=compactor)
