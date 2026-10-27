"""Regression: vectors from two embedding models must never be mixed in one index, and a cache must not cross models."""
from contextrag import factory
from contextrag.core.cache import CachedEmbedder, InMemoryCache
from contextrag.core.embeddings import HashingEmbedder

TEXT = "El plazo de prescripción de la acción es de cuatro años contados desde la notificación."


class OtherModel(HashingEmbedder):
    """Same dimension, a different vector space (stands in for a second embedding model)."""

    def __init__(self):
        super().__init__(dim=512)
        self.model_id = "other-model-v2:512"

    def embed(self, texts):
        return [[-x for x in v] for v in super().embed(texts)]


def test_a_changed_embedding_model_forces_reembedding_of_an_unchanged_document():
    r = factory.offline()
    first = r.ingestor.ingest_text(TEXT, tenant="t", doc_id="d.md", source="d.md")
    assert first["unchanged"] is False
    r.ingestor.embedder = OtherModel()
    again = r.ingestor.ingest_text(TEXT, tenant="t", doc_id="d.md", source="d.md")
    assert again["unchanged"] is False  # before the fix this was True: old-model vectors stayed in the index


def test_the_same_model_still_skips_unchanged_documents():
    r = factory.offline()
    r.ingestor.ingest_text(TEXT, tenant="t", doc_id="d.md", source="d.md")
    assert r.ingestor.ingest_text(TEXT, tenant="t", doc_id="d.md", source="d.md")["unchanged"] is True


def test_the_index_holds_only_the_new_model_after_reembedding():
    r = factory.offline()
    r.ingestor.ingest_text(TEXT, tenant="t", doc_id="d.md", source="d.md")
    old_vec = r.index.store.dense(HashingEmbedder().embed([TEXT])[0], "t", frozenset(), 1)[0][1]
    r.ingestor.embedder = OtherModel()
    r.ingestor.ingest_text(TEXT, tenant="t", doc_id="d.md", source="d.md")
    new_score = r.index.store.dense(HashingEmbedder().embed([TEXT])[0], "t", frozenset(), 1)[0][1]
    assert old_vec > 0.99 and new_score < 0  # the old query vector no longer matches the re-embedded chunk


def test_the_cache_keys_vectors_by_model_identity_not_by_a_default_name():
    cache = InMemoryCache()
    CachedEmbedder(HashingEmbedder(), cache).embed([TEXT])
    other = CachedEmbedder(OtherModel(), cache)
    assert other.model_id == "other-model-v2:512"
    v = other.embed([TEXT])[0]
    assert v == OtherModel().embed([TEXT])[0]  # computed by the other model, not served from the first model's entry


def test_a_wrapped_embedder_keeps_its_identity():
    assert CachedEmbedder(HashingEmbedder(64), InMemoryCache()).model_id == "hashing-v1:64"
