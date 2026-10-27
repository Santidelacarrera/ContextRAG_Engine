"""Reindexing must not stop readers while the expensive part runs, and a reader must never see two versions at once.

Regression for a claim that was made without measurement (\"reindexing blocks reads\"). Measured: the slow part of an ingest
is the embedding, which happens before the index lock is taken, so reads continue during it.
"""
import re
import threading
import time

from contextrag import Status, factory
from contextrag.core.embeddings import HashingEmbedder
from contextrag.eval.retrieval_bench import TENANT, load_dataset

Q = "¿Cuál es el plazo de prescripción de la acción?"
OLD = "El plazo de prescripción de la acción es de cuatro años contados desde la notificación."
NEW = "El plazo de prescripción de la acción es de dos años contados desde la notificación."


class SlowIngestEmbedder(HashingEmbedder):
    def embed(self, texts):
        time.sleep(0.15)  # stands in for a remote embedding call during ingestion
        return super().embed(texts)


def test_reads_continue_while_a_slow_ingest_is_embedding():
    data, _ = load_dataset()
    rag = factory.offline()
    rag.cache = None
    rag.ingestor.cache = None
    for d in data["documents"]:
        rag.ingestor.ingest_text(d["text"], tenant=TENANT, doc_id=d["doc_id"], source=d["doc_id"])
    rag.ingestor.embedder = SlowIngestEmbedder()  # only ingestion is slow; queries keep the fast embedder
    stop, latencies = threading.Event(), []

    def reader():
        i = 0
        while not stop.is_set():
            t0 = time.perf_counter()
            rag.ask(data["queries"][i % len(data["queries"])]["question"], tenant=TENANT)
            latencies.append((time.perf_counter() - t0) * 1000)
            i += 1

    def writer():
        for d in data["documents"][:3]:
            rag.ingestor.ingest_text(d["text"], tenant=TENANT, doc_id=d["doc_id"], source=d["doc_id"], force=True)
        stop.set()

    r = threading.Thread(target=reader)
    r.start()
    time.sleep(0.1)
    w = threading.Thread(target=writer)
    w.start()
    w.join(30)
    r.join(30)
    assert len(latencies) >= 20  # reads completed while the three slow ingests ran
    assert max(latencies) < 150  # and none waited for the embedding (150 ms per batch)


def test_a_reader_never_sees_a_mix_of_two_versions_during_replacement():
    rag = factory.offline()
    rag.ingestor.ingest_text(OLD, tenant="t", doc_id="d.md", source="d.md")
    stop, mixed = threading.Event(), []

    def writer():
        for i in range(60):
            rag.ingestor.ingest_text(NEW if i % 2 else OLD, tenant="t", doc_id="d.md", source="d.md", force=True)
        stop.set()

    def reader():
        while not stop.is_set():
            a = rag.ask(Q, tenant="t")
            # whole words: "contados" contains the substring "dos" and made a single-version answer look mixed
            has_old, has_new = bool(re.search(r"\bcuatro\b", a.text)), bool(re.search(r"\bdos\b", a.text))
            if a.status is Status.VERIFIED and has_old == has_new:
                mixed.append(a.text)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert mixed == []
