"""Component wiring. ``offline()`` needs no models or services; ``from_env()`` picks
backends from environment variables (used by the API server and docker-compose)."""
from __future__ import annotations

import os

from .audit import InMemoryAudit
from .core.cache import CachedEmbedder, InMemoryCache
from .core.config import Settings
from .core.embeddings import HashingEmbedder
from .generation.llm import ExtractiveLLM
from .pipeline import ContextRAG
from .retrieval.hybrid_search import HybridIndex
from .retrieval.reranker import LexicalReranker
from .retrieval.vector_store import InMemoryVectorStore
from .verification.nli_verifier import LexicalEntailment


def offline(settings: Settings | None = None, llm=None) -> ContextRAG:
    """Self-contained stack: hashing embeddings, in-memory stores, lexical rerank/NLI, extractive LLM."""
    return ContextRAG(index=HybridIndex(InMemoryVectorStore()), embedder=HashingEmbedder(),
                      reranker=LexicalReranker(), llm=llm or ExtractiveLLM(), nli=LexicalEntailment(),
                      audit=InMemoryAudit(), cache=InMemoryCache(), settings=settings)


def from_env() -> ContextRAG:
    """Backends are chosen by env vars:

    EMBEDDER=hashing|bge (EMBED_MODEL)   RERANKER=lexical|cross-encoder|cohere (COHERE_API_KEY)
    NLI=lexical|deberta                  LLM=extractive|anthropic|ollama (LLM_MODEL, OLLAMA_HOST)
    QDRANT_URL (else embedded in-memory)  POSTGRES_DSN (else in-memory audit)  REDIS_URL (else in-memory cache)
    MASTER_KEY (enables per-tenant encryption of chunk text at rest)  CONTEXTRAG_* (see Settings)
    """
    env = os.getenv
    cfg = Settings.from_env()

    cache = InMemoryCache()
    if env("REDIS_URL"):
        from .core.cache import RedisCache

        cache = RedisCache(env("REDIS_URL"))

    if env("EMBEDDER", "hashing") == "bge":
        from .core.embeddings import SentenceTransformerEmbedder

        name = env("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
        embedder = CachedEmbedder(SentenceTransformerEmbedder(name), cache, name)
    else:
        embedder = HashingEmbedder()

    from .core.security import TenantCipher

    cipher = TenantCipher(env("MASTER_KEY").encode()) if env("MASTER_KEY") else None
    if env("QDRANT_URL") or cipher:
        from qdrant_client import QdrantClient

        from .retrieval.vector_store import QdrantVectorStore

        client = QdrantClient(url=env("QDRANT_URL")) if env("QDRANT_URL") else None
        store = QdrantVectorStore(embedder.dim, client=client, cipher=cipher)
    else:
        store = InMemoryVectorStore()

    rer = env("RERANKER", "lexical")
    if rer == "cohere":
        from .retrieval.reranker import CohereReranker

        reranker = CohereReranker(env("COHERE_API_KEY"))
    elif rer == "cross-encoder":
        from .retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(env("RERANK_MODEL", "BAAI/bge-reranker-large"))
    else:
        reranker = LexicalReranker()

    if env("NLI", "lexical") == "deberta":
        from .verification.nli_verifier import DebertaNLI

        nli = DebertaNLI()
    else:
        nli = LexicalEntailment()

    llm_kind = env("LLM", "extractive")
    if llm_kind == "anthropic":
        from .generation.llm import AnthropicLLM

        llm = AnthropicLLM(env("LLM_MODEL"))
    elif llm_kind == "ollama":
        from .generation.llm import OllamaLLM

        llm = OllamaLLM(env("LLM_MODEL", "llama3"), env("OLLAMA_HOST", "http://localhost:11434"))
    else:
        llm = ExtractiveLLM()

    if env("POSTGRES_DSN"):
        from .audit import PostgresAudit

        audit = PostgresAudit(env("POSTGRES_DSN"), embedder.dim)
    else:
        audit = InMemoryAudit()

    return ContextRAG(index=HybridIndex(store), embedder=embedder, reranker=reranker, llm=llm, nli=nli,
                      audit=audit, cache=cache, settings=cfg)
