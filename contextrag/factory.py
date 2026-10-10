"""Wiring helpers. ``offline()`` needs no models or services."""
from __future__ import annotations

from .audit import InMemoryAudit
from .config import Settings
from .embeddings import HashingEmbedder
from .llm import ExtractiveLLM
from .nli import LexicalEntailment
from .pipeline import ContextRAG
from .rerank import LexicalReranker
from .store import InMemoryStore


def offline(settings: Settings | None = None, llm=None) -> ContextRAG:
    return ContextRAG(store=InMemoryStore(), embedder=HashingEmbedder(), reranker=LexicalReranker(),
                      llm=llm or ExtractiveLLM(), nli=LexicalEntailment(), audit=InMemoryAudit(), settings=settings)


def production(*, qdrant_url: str, anthropic_model: str | None = None, postgres_dsn: str | None = None,
               cohere_key: str | None = None, master_key: bytes | None = None,
               embed_model: str = "BAAI/bge-large-en-v1.5", settings: Settings | None = None) -> ContextRAG:
    from qdrant_client import QdrantClient

    from .audit import PostgresAudit
    from .embeddings import SentenceTransformerEmbedder
    from .llm import AnthropicLLM
    from .nli import DebertaNLI
    from .qdrant_store import QdrantStore
    from .rerank import CohereReranker, CrossEncoderReranker
    from .security import TenantCipher

    emb = SentenceTransformerEmbedder(embed_model)
    store = QdrantStore(emb.dim, client=QdrantClient(url=qdrant_url), cipher=TenantCipher(master_key) if master_key else None)
    return ContextRAG(
        store=store, embedder=emb,
        reranker=CohereReranker(cohere_key) if cohere_key else CrossEncoderReranker(),
        llm=AnthropicLLM(anthropic_model), nli=DebertaNLI(),
        audit=PostgresAudit(postgres_dsn) if postgres_dsn else None, settings=settings,
    )
