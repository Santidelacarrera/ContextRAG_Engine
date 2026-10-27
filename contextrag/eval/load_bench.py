"""Reproducible load benchmark for the in-process pipeline: throughput and latency percentiles at several concurrency levels.

What this measures: how the retrieval-and-verification pipeline behaves when several threads call it at once, on the
offline stack (hashing embedder, lexical reranker and NLI, extractive generator), in one Python process on one machine.
Query order is fixed by a seed, the dataset is the versioned synthetic corpus, and the answer cache is off by default so that every
request does the full work; ``with_cache=True`` measures the cached path separately.

What it does NOT measure, and must not be quoted as: capacity of a deployment. There is no network, no real model, no
database, no HTTP layer, and Python threads share one interpreter lock, so throughput will not grow with concurrency the way
a deployment with separate workers would. Numbers depend on the machine and are for comparing two runs on the same
machine (for example before and after a change), never for promising a number of users.

Latency percentiles are reported only with at least 100 samples (p99 needs more to mean anything); with fewer they are null.
"""
from __future__ import annotations

import os
import platform
import random
import resource
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..core.models import Status
from .bench_corpus import DATA_PATH
from .retrieval_bench import TENANT, latency_summary, load_dataset

CAVEAT = ("In-process, offline stack, one machine, threads sharing one interpreter lock. A comparison tool for two runs on "
          "the same machine; not the capacity of a deployment and not evidence about production models.")


def _worker_setup(dataset: str | Path = DATA_PATH, with_cache: bool = False):
    from ..factory import offline

    data, sha = load_dataset(dataset)
    rag = offline()
    if not with_cache:
        # offline() enables an in-memory answer cache. Without this line the "read-only" figures would mostly measure cache
        # hits, which is what an earlier run of this benchmark did before the cache was turned off here.
        rag.cache = None
        rag.ingestor.cache = None
    for d in data["documents"]:
        rag.ingestor.ingest_text(d["text"], tenant=TENANT, doc_id=d["doc_id"], source=d["doc_id"])
    return rag, data, sha


def run_level(rag: Any, questions: list[str], concurrency: int, requests: int, seed: int, ingest_every: int = 0,
              documents: list[dict] | None = None) -> dict[str, Any]:
    # seeded on purpose (reproducible evaluation data, not security-sensitive)
    rng = random.Random(seed)  # nosec B311
    plan = [rng.choice(questions) for _ in range(requests)]
    latencies: list[float] = []
    outcomes: dict[str, int] = {"verified": 0, "fallback": 0, "error": 0}
    lock = threading.Lock()
    ingests = {"n": 0, "failed": 0}

    def one(i: int) -> None:
        t0 = time.perf_counter()
        try:
            if ingest_every and documents and i % ingest_every == 0:  # concurrent writes against the readers
                d = documents[i % len(documents)]
                rag.ingestor.ingest_text(d["text"], tenant=TENANT, doc_id=d["doc_id"], source=d["doc_id"], force=True)
                with lock:
                    ingests["n"] += 1
                return
            status = rag.ask(plan[i], tenant=TENANT).status
            key = "verified" if status is Status.VERIFIED else "fallback"
        except Exception:
            key = "error"
            if ingest_every and i % ingest_every == 0:
                with lock:
                    ingests["failed"] += 1
        with lock:
            outcomes[key] += 1
            if key != "error":
                latencies.append((time.perf_counter() - t0) * 1000)

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, range(requests)))
    wall = time.perf_counter() - start
    done = sum(outcomes.values())
    return {"concurrency": concurrency, "requests": requests, "wall_seconds": round(wall, 3),
            "throughput_per_second": round(done / wall, 1) if wall else None, "outcomes": outcomes,
            "ingests": ingests if ingest_every else None,
            "latency": latency_summary(latencies) if latencies else None}


def run_load(levels: tuple[int, ...] = (1, 4, 8, 16), requests: int = 400, seed: int = 7,
             mixed_ingest_every: int = 10, dataset: str | Path = DATA_PATH, with_cache: bool = False) -> dict[str, Any]:
    if not levels or any(n < 1 for n in levels) or requests < 1:
        raise ValueError("levels must be positive integers and requests >= 1")
    rag, data, sha = _worker_setup(dataset, with_cache)
    try:
        questions = [q["question"] for q in data["queries"]]
        run_level(rag, questions, 1, min(20, requests), seed)  # warm-up, discarded
        read_only = [run_level(rag, questions, n, requests, seed) for n in levels]
        mixed = [run_level(rag, questions, n, requests, seed, mixed_ingest_every, data["documents"]) for n in levels] \
            if mixed_ingest_every else []
        return {
            "caveat": CAVEAT,
            "dataset": {"sha256": sha, "documents": len(data["documents"]), "queries": len(questions)},
            "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpus": os.cpu_count()},
            "seed": seed,
            "answer_cache": with_cache,
            "read_only": read_only,
            "mixed_read_write": {"ingest_every": mixed_ingest_every, "levels": mixed} if mixed_ingest_every else None,
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),  # whole process, Linux
            "not_measured": ["HTTP and serialisation overhead", "network and database latency", "real models",
                             "multi-process deployment", "sustained load over time", "memory growth over hours"],
        }
    finally:
        rag.close()
