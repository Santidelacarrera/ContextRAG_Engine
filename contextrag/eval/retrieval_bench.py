"""Retrieval and grounding benchmark over the versioned synthetic corpus.

Four retrieval strategies run through the real pipeline components, ranked at document level:

* ``bm25``: sparse retrieval alone.
* ``dense``: vector retrieval alone.
* ``hybrid``: BM25 + dense fused with reciprocal rank fusion, no reranking.
* ``hybrid_rerank``: the production retrieval path (fusion, then the configured reranker).

Metrics are Recall@k, Precision@k, MRR and nDCG@k with binary relevance, plus per-query latency percentiles. A second
section runs the full ``ask`` path and reports answer correctness, citation precision, verified-but-wrong rate, and
abstention on questions the corpus cannot answer.

What this does NOT show. The offline stack uses a hashing embedder (lexical, not semantic), a lexical reranker, a
lexical NLI model and an extractive generator. The numbers measure how the pipeline behaves on this corpus. They say
nothing about retrieval quality with production models or real documents. Token cost and concurrent throughput are not
measured here and are reported as ``not_measured``.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .bench_corpus import DATA_PATH

BASELINE_PATH = DATA_PATH.with_name("retrieval_bench_baseline.json")

KS = (1, 3, 5, 10)
STRATEGIES = ("bm25", "dense", "hybrid", "hybrid_rerank")
TENANT = "bench"
NOT_MEASURED = ["token consumption and cost per query (no paid model in the offline stack)",
                "concurrent throughput", "retrieval quality with semantic embeddings and a trained reranker",
                "real-document retrieval"]


# ---- metrics (binary relevance, document level) ---------------------------------------------------------

def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / k


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def latency_summary(ms: Sequence[float]) -> dict[str, Any]:
    """p99 is reported only with at least 100 samples; below that it is the maximum in disguise."""
    return {"n": len(ms), "p50_ms": round(percentile(ms, 50), 3), "p95_ms": round(percentile(ms, 95), 3),
            "p99_ms": round(percentile(ms, 99), 3) if len(ms) >= 100 else None, "max_ms": round(max(ms), 3)}


def _docs_in_order(children: Sequence[Any]) -> list[str]:
    seen: dict[str, None] = {}
    for c in children:
        seen.setdefault(c.doc_id, None)
    return list(seen)


# ---- dataset and runners ----------------------------------------------------------------------------------

def load_dataset(path: str | Path = DATA_PATH) -> tuple[dict, str]:
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _build_rag(data: dict, make_rag: Callable[[], Any]) -> tuple[Any, dict]:
    rag = make_rag()
    t0 = time.perf_counter()
    chunks = 0
    for d in data["documents"]:
        info = rag.ingestor.ingest_text(d["text"], tenant=TENANT, doc_id=d["doc_id"], source=d["doc_id"])
        chunks += info["children"]
    seconds = time.perf_counter() - t0
    return rag, {"documents": len(data["documents"]), "chunks": chunks, "seconds": round(seconds, 4),
                 "documents_per_second": round(len(data["documents"]) / seconds, 1) if seconds else None,
                 "failures": 0}


def _rankers(rag: Any) -> dict[str, Callable[[str], list[str]]]:
    cfg = rag.cfg
    groups: frozenset[str] = frozenset()

    def bm25(q: str) -> list[str]:
        return _docs_in_order([c for c, _ in rag.index.bm25.search(q, TENANT, groups, cfg.sparse_k)])

    def dense(q: str) -> list[str]:
        qvec = rag.embedder.embed([q])[0]
        return _docs_in_order([c for c, _ in rag.index.store.dense(qvec, TENANT, groups, cfg.dense_k)])

    def hybrid(q: str) -> list[str]:
        return _docs_in_order([c for c, _ in rag.retriever.search_children(q, TENANT, groups)])

    def hybrid_rerank(q: str) -> list[str]:
        ctxs = rag.retriever.retrieve(q, TENANT, groups, top_k=max(KS))
        return list(dict.fromkeys(c.parent.doc_id for c in ctxs))

    return {"bm25": bm25, "dense": dense, "hybrid": hybrid, "hybrid_rerank": hybrid_rerank}


def retrieval_section(rag: Any, queries: Sequence[dict], repeats: int = 1) -> dict[str, Any]:
    answerable = [q for q in queries if q["relevant"]]
    out: dict[str, Any] = {}
    for name, rank in _rankers(rag).items():
        per_kind: dict[str, dict[str, list[float]]] = {}
        latencies: list[float] = []
        for q in answerable:
            relevant = set(q["relevant"])
            ranked: list[str] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                ranked = rank(q["question"])
                latencies.append((time.perf_counter() - t0) * 1000)
            for kind in ("all", q["kind"]):
                m = per_kind.setdefault(kind, {})
                m.setdefault("mrr", []).append(reciprocal_rank(ranked, relevant))
                for k in KS:
                    m.setdefault(f"recall@{k}", []).append(recall_at_k(ranked, relevant, k))
                    m.setdefault(f"precision@{k}", []).append(precision_at_k(ranked, relevant, k))
                    m.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(ranked, relevant, k))
        out[name] = {
            "queries": {kind: len(next(iter(m.values()))) for kind, m in per_kind.items()},
            "metrics": {kind: {k: round(sum(v) / len(v), 4) for k, v in m.items()} for kind, m in per_kind.items()},
            "latency": latency_summary(latencies),
        }
    return out


def grounding_section(rag: Any, queries: Sequence[dict]) -> dict[str, Any]:
    from ..core.models import Status

    answerable = [q for q in queries if q["relevant"] and "expect" in q]
    unanswerable = [q for q in queries if not q["relevant"]]
    verified = correct = verified_wrong = covered = 0
    precisions: list[float] = []
    latencies: list[float] = []
    for q in answerable:
        t0 = time.perf_counter()
        a = rag.ask(q["question"], tenant=TENANT)
        latencies.append((time.perf_counter() - t0) * 1000)
        if a.status is not Status.VERIFIED:
            continue
        verified += 1
        ok = q["expect"].lower() in a.text.lower()
        correct += ok
        verified_wrong += not ok
        cited = [v.evidence.doc_id for v in a.claims if v.evidence]  # the evidence each claim actually rests on
        if cited:
            precisions.append(sum(d in set(q["relevant"]) for d in cited) / len(cited))
        covered += bool(a.claims) and all(v.evidence for v in a.claims)
    abstained = 0
    for q in unanswerable:
        t0 = time.perf_counter()
        a = rag.ask(q["question"], tenant=TENANT)
        latencies.append((time.perf_counter() - t0) * 1000)
        abstained += a.status is not Status.VERIFIED
    n = len(answerable)
    return {
        "answerable": {"queries": n, "verified_rate": round(verified / n, 4), "correct_rate": round(correct / n, 4),
                       "abstained_rate": round(1 - verified / n, 4),
                       "verified_wrong_rate": round(verified_wrong / verified, 4) if verified else None,
                       "citation_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
                       "citation_coverage": round(covered / verified, 4) if verified else None},
        "unanswerable": {"queries": len(unanswerable),
                         "abstention_rate": round(abstained / len(unanswerable), 4),
                         "answered_as_supported_rate": round(1 - abstained / len(unanswerable), 4)},
        "end_to_end_latency": latency_summary(latencies),
    }


def run_benchmark(*, dataset: str | Path = DATA_PATH, make_rag: Callable[[], Any] | None = None,
                  repeats: int = 1) -> dict[str, Any]:
    from ..factory import offline

    data, sha = load_dataset(dataset)
    rag, indexing = _build_rag(data, make_rag or offline)
    try:
        queries = data["queries"]
        return {
            "dataset": {"version": data["version"], "seed": data["seed"], "synthetic": data["synthetic"], "sha256": sha,
                        "documents": len(data["documents"]), "queries": len(queries)},
            "components": {"embedder": type(rag.embedder).__name__, "reranker": type(rag.reranker).__name__,
                           "nli": type(rag.verifier.nli).__name__, "llm": type(rag.llm).__name__},
            "settings": asdict(rag.cfg),
            "repeats": repeats,
            "caveat": ("Offline stack: hashing embedder, lexical reranker and NLI, extractive generator. These figures "
                       "describe pipeline behaviour on a synthetic corpus, not production retrieval quality."),
            "indexing": indexing,
            "retrieval": retrieval_section(rag, queries, repeats),
            "grounding": grounding_section(rag, queries),
            "not_measured": NOT_MEASURED,
        }
    finally:
        rag.close()


def regressions(report: dict, baseline: dict, tolerance: float = 0.0) -> list[str]:
    """Metrics that fell below the recorded floor. ``baseline`` maps ``"strategy.metric"`` to a minimum, and
    ``"grounding.x"`` entries to a minimum (or, for ``*_max`` keys, a maximum)."""
    problems = []
    for key, floor in baseline["retrieval_floor"].items():
        strategy, metric = key.split(".", 1)
        value = report["retrieval"][strategy]["metrics"]["all"][metric]
        if value < floor - tolerance:
            problems.append(f"{key}: {value:.4f} < baseline {floor:.4f}")
    g = report["grounding"]
    for key, floor in baseline.get("grounding_floor", {}).items():
        section, metric = key.split(".", 1)
        value = g[section][metric]
        if value is None or value < floor - tolerance:
            problems.append(f"grounding.{key}: {value} < baseline {floor:.4f}")
    for key, ceiling in baseline.get("grounding_ceiling", {}).items():
        section, metric = key.split(".", 1)
        value = g[section][metric]
        if value is not None and value > ceiling + tolerance:
            problems.append(f"grounding.{key}: {value:.4f} > ceiling {ceiling:.4f}")
    return problems
