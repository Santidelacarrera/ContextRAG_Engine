"""``python -m contextrag {serve,ingest,ask,eval}``. Backends come from environment variables
(see ``contextrag.factory.from_env``); with none set everything runs in memory."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="contextrag", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    i = sub.add_parser("ingest", help="index a file or directory (needs a persistent backend to be useful)")
    i.add_argument("path", type=Path)
    i.add_argument("--tenant", required=True)
    i.add_argument("--acl", default="", help="comma-separated groups allowed to read the documents")

    q = sub.add_parser("ask", help="ask a question, optionally ingesting --docs first")
    q.add_argument("question")
    q.add_argument("--tenant", required=True)
    q.add_argument("--docs", type=Path, help="file or directory to ingest before asking")
    q.add_argument("--groups", default="")

    k = sub.add_parser("keygen", help="create an API key bound to a tenant/role; prints the key once + the keys-file entry")
    k.add_argument("--id", required=True, help="label that shows up in the audit log")
    k.add_argument("--tenant", default="", help="required unless --role ops")
    k.add_argument("--role", choices=["reader", "admin", "ops"], default="reader")
    k.add_argument("--groups", default="", help="comma-separated ACL groups")
    k.add_argument("--add-to", type=Path, help="keys file to create/update (CONTEXTRAG_KEYS_FILE)")
    k.add_argument("--expires-in-days", type=float, help="the key stops working after this many days (recommended)")

    x = sub.add_parser("erase", help="irreversibly delete ALL data of a tenant (documents, vectors, cache, audit)")
    x.add_argument("--tenant", required=True)
    x.add_argument("--keep-audit", action="store_true", help="legal hold: keep the audit records")
    x.add_argument("--yes", action="store_true", help="required: confirms the erasure")

    pa = sub.add_parser("audit-purge", help="retention: delete audit records older than N days (all tenants)")
    pa.add_argument("--older-than-days", type=float, required=True)

    rt = sub.add_parser("retention", help="keep the last N calendar months of audit data (admin DSN; run from cron)")
    rt.add_argument("--months", type=int, required=True, help="calendar months to keep, including the current one")
    rt.add_argument("--dry-run", action="store_true", help="report what would be removed, change nothing")

    sub.add_parser("memory-migrate", help="encrypt conversation rows written before MEMORY_DB was encrypted "
                                          "(needs MEMORY_DB and MASTER_KEY; idempotent, safe to re-run)")

    bn = sub.add_parser("bench", help="retrieval + grounding benchmark on the versioned synthetic corpus; with "
                                      "--check-baseline exit 1 if a metric fell below the recorded floor")
    bn.add_argument("--out", help="write the full JSON report here")
    bn.add_argument("--dataset", help="alternative dataset file in the same format")
    bn.add_argument("--repeats", type=int, default=1, help="run each retrieval query this many times for latency")
    bn.add_argument("--check-baseline", action="store_true")

    ld = sub.add_parser("loadtest", help="in-process load benchmark (offline stack): latency percentiles and throughput at "
                                         "several concurrency levels; a comparison tool, not a capacity claim")
    ld.add_argument("--levels", default="1,4,8,16", help="comma-separated concurrency levels")
    ld.add_argument("--requests", type=int, default=400, help="requests per level")
    ld.add_argument("--seed", type=int, default=7)
    ld.add_argument("--mixed-ingest-every", type=int, default=10, help="every Nth request is a re-ingest (0 = read-only only)")
    ld.add_argument("--with-cache", action="store_true", help="keep the answer cache on (measures the cached path)")
    ld.add_argument("--out", help="write the full JSON report here")

    rk = sub.add_parser("rotate-key", help="re-encrypt data still under a previous master key (needs MASTER_KEY as the new "
                                           "primary and MASTER_KEY_PREVIOUS; covers MEMORY_DB and QDRANT_URL)")
    rk.add_argument("--dry-run", action="store_true", help="count what would be re-encrypted and change nothing")

    rg = sub.add_parser("ragas", help="LLM-judge evaluation (faithfulness, answer relevancy, context recall); needs "
                                      "RAGAS_PROVIDER and the provider key. Exit 1 if any judge call failed")
    rg.add_argument("--cases", required=True, help="JSON of {documents, cases}; cases need id, question, reference")
    rg.add_argument("--out", required=True, help="where to write the JSON report")
    rg.add_argument("--metrics", default="faithfulness,answer_relevancy,context_recall")
    rg.add_argument("--tenant", default="ragas")

    cal = sub.add_parser("calibrate", help="calibrate chunk sizes (1024/2000 x 256/400), RERANK_MIN_SCORE and the NLI "
                                           "threshold; writes a report, never touches production configuration")
    cal.add_argument("--cases", help="JSON of {documents, cases} with labelled 'doc' per answerable case")
    cal.add_argument("--synthetic", action="store_true", help="use the deterministic synthetic long corpus instead")
    cal.add_argument("--seed", type=int, default=7)
    cal.add_argument("--verifier-cases", help="JSON of statement-level labels {premise, claim, label}")
    cal.add_argument("--target-precision", type=float, default=0.9)
    cal.add_argument("--out", required=True, help="JSON report path")
    cal.add_argument("--proposal", help="optional path for candidate values to review (not applied anywhere)")

    e = sub.add_parser("eval", help="run the built-in (or custom) evaluation; exit 1 if anything is unsafe")
    e.add_argument("--verifier-cases", help="JSON file of {premise, claim, label, category, id}")
    e.add_argument("--rag-cases", help="JSON file of {documents, cases}")
    e.add_argument("--sweep", action="store_true", help="also print false-support/recall for a range of thresholds")
    e.add_argument("--json", action="store_true", help="print machine-readable results")

    args = ap.parse_args(argv)
    from .core.redact import install_log_redaction

    install_log_redaction()
    logging.basicConfig(level=logging.WARNING if args.cmd in ("eval", "bench", "loadtest") else logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    csv = lambda v: tuple(x for x in v.split(",") if x)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("contextrag.api.server:app_factory", factory=True, host=args.host, port=args.port)
        return 0

    if args.cmd == "keygen":
        import os

        from .api.auth import KeyStore, new_entry

        from datetime import datetime, timedelta, timezone

        expires = None
        if args.expires_in_days is not None:
            if args.expires_in_days <= 0:
                print("keygen: --expires-in-days must be positive", file=sys.stderr)
                return 2
            expires = (datetime.now(timezone.utc) + timedelta(days=args.expires_in_days)).isoformat(timespec="seconds")
        key, digest, entry = new_entry(args.id, args.tenant, args.role, csv(args.groups), expires_at=expires)
        KeyStore({digest: entry})  # validates the entry (e.g. tenant required for non-ops roles)
        if args.add_to:
            entries = json.loads(args.add_to.read_text()) if args.add_to.exists() else {}
            entries[digest] = entry
            args.add_to.write_text(json.dumps(entries, indent=2))
            os.chmod(args.add_to, 0o600)
        else:
            print(json.dumps({digest: entry}, indent=2), file=sys.stderr)
        print(key)  # shown once; only its SHA-256 is stored
        return 0

    from . import factory

    if args.cmd == "ingest":
        rag = factory.from_env()
        print(json.dumps(rag.ingestor.ingest_path(args.path, tenant=args.tenant, acl=csv(args.acl)),
                         ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "erase":
        if not args.yes:
            print("refusing to erase without --yes", file=sys.stderr)
            return 2
        print(json.dumps(factory.from_env().erase_tenant(args.tenant, include_audit=not args.keep_audit)))
        return 0

    if args.cmd == "audit-purge":
        print(json.dumps({"purged": factory.from_env().purge_audit(args.older_than_days)}))
        return 0

    if args.cmd == "loadtest":
        return _run_loadtest(args)

    if args.cmd == "rotate-key":
        return _run_rotate_key(args)

    if args.cmd == "bench":
        return _run_bench(args)

    if args.cmd == "ragas":
        return _run_ragas(args)

    if args.cmd == "calibrate":
        return _run_calibrate(args)

    if args.cmd == "memory-migrate":
        import os

        from .core.security import TenantCipher
        from .memory.store import ConversationStore

        db_path, master_key = os.getenv("MEMORY_DB"), os.getenv("MASTER_KEY")
        if not db_path or not master_key:
            print("MEMORY_DB and MASTER_KEY are required", file=sys.stderr)
            return 2
        store = ConversationStore(db_path, cipher=TenantCipher(master_key.encode()))
        try:
            print(json.dumps({"encrypted": store.migrate_plaintext()}))
        finally:
            store.close()
        return 0

    if args.cmd == "retention":
        import os

        import psycopg

        from . import retention

        dsn = os.getenv("POSTGRES_ADMIN_DSN") or os.getenv("POSTGRES_DSN")
        if not dsn:
            print("POSTGRES_ADMIN_DSN (or POSTGRES_DSN) is required", file=sys.stderr)
            return 2
        with psycopg.connect(dsn, autocommit=True) as conn:
            results = retention.run_retention(conn, args.months, dry_run=args.dry_run)
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
        return 0

    if args.cmd == "ask":
        rag = factory.from_env()
        if args.docs:
            rag.ingestor.ingest_path(args.docs, tenant=args.tenant)
        a = rag.ask(args.question, tenant=args.tenant, groups=frozenset(csv(args.groups)))
        print(json.dumps(a.to_response(), ensure_ascii=False, indent=2))
        return 0 if a.status.value == "verified" else 2

    from .eval.harness import eval_rag, eval_verifier, load_json, sweep_thresholds

    rag = factory.from_env()
    vr = eval_verifier(rag.verifier.nli, rag.cfg.verification_threshold,
                       load_json("", args.verifier_cases) if args.verifier_cases else None)
    rr = eval_rag(factory.from_env, load_json("", args.rag_cases) if args.rag_cases else None)
    vcases = load_json("", args.verifier_cases) if args.verifier_cases else None
    sweep = sweep_thresholds(rag.verifier.nli, vcases) if args.sweep else None
    if args.json:
        out = {"verifier": vr.to_dict(), "rag": rr.to_dict()}
        if sweep:
            out["sweep"] = sweep
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(f"verifier  cases={vr.total}  false_support_rate={vr.false_support_rate:.1%}  recall={vr.recall:.1%}")
        for c, acc in sorted(vr.by_category.items()):
            print(f"  {c:<14} {acc:.0%}")
        print(f"rag       cases={rr.total}  answer={rr.answer_rate:.0%}  correct={rr.correct_rate:.0%}  "
              f"abstain={rr.abstention_rate:.0%}  unsafe={rr.unsafe_rate:.0%}  hit@k={rr.retrieval_hit_rate:.0%}  "
              f"mrr={rr.mrr:.2f}")
        for label, ids in (("false supports", vr.false_supports), ("unsafe answers", rr.unsafe),
                           ("missed answers", rr.failures)):
            if ids:
                print(f"  {label}: {', '.join(ids)}")
        if sweep:
            print("threshold  false_support  recall")
            for r in sweep["rows"]:
                print(f"  {r['threshold']:<8} {r['false_support_rate']:>12.1%} {r['recall']:>8.1%}")
            print(f"  recommended threshold (lowest with zero false supports): {sweep['recommended']}")
    return 1 if (vr.false_supports or rr.unsafe) else 0


def _run_loadtest(args) -> int:
    from .eval.load_bench import run_load

    try:
        levels = tuple(int(x) for x in args.levels.split(",") if x)
        report = run_load(levels, args.requests, args.seed, args.mixed_ingest_every, with_cache=args.with_cache)
    except ValueError as exc:
        print(f"loadtest: {exc}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["caveat"])
    for title, rows in (("read-only", report["read_only"]),
                        ("mixed read/write", (report["mixed_read_write"] or {}).get("levels", []))):
        print(f"\n{title}")
        print(f"{'conc':>5s} {'req/s':>8s} {'p50 ms':>8s} {'p95 ms':>8s} {'p99 ms':>8s} {'errors':>7s}")
        for r in rows:
            lat = r["latency"] or {}
            p99 = lat.get("p99_ms")
            print(f"{r['concurrency']:5d} {r['throughput_per_second']:8.1f} {lat.get('p50_ms', float('nan')):8.2f} "
                  f"{lat.get('p95_ms', float('nan')):8.2f} {('%.2f' % p99) if p99 is not None else 'n<100':>8s} "
                  f"{r['outcomes']['error']:7d}")
    print(f"\npeak RSS (whole process): {report['peak_rss_mb']} MB")
    return 1 if any(r["outcomes"]["error"] for r in report["read_only"]) else 0


def _run_rotate_key(args) -> int:
    import os

    from .core.security import TenantCipher

    from .core.security import load_master_keys

    try:
        key, previous = load_master_keys()
    except (ValueError, OSError) as exc:  # the message names the variable, never the key
        print(f"rotate-key: {exc}", file=sys.stderr)
        return 2
    if not key or not previous:
        print("rotate-key: set MASTER_KEY (or MASTER_KEY_FILE) for the new key and MASTER_KEY_PREVIOUS (or "
              "MASTER_KEY_PREVIOUS_FILE) for the key(s) being retired", file=sys.stderr)
        return 2
    memory_path, qdrant_url = os.getenv("MEMORY_DB"), os.getenv("QDRANT_URL")
    if not memory_path and not qdrant_url:
        print("rotate-key: nothing to rotate; set MEMORY_DB and/or QDRANT_URL", file=sys.stderr)
        return 2
    cipher = TenantCipher(key, previous)
    result: dict[str, object] = {"dry_run": args.dry_run}
    try:
        if memory_path:
            from .memory.store import ConversationStore

            store = ConversationStore(memory_path, cipher=cipher)
            try:
                result["memory"] = store.rekey(dry_run=args.dry_run)
            finally:
                store.close()
        if qdrant_url:
            from qdrant_client import QdrantClient

            from .retrieval.vector_store import QdrantVectorStore

            client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY") or None)
            if not client.collection_exists("contextrag"):
                print("rotate-key: the Qdrant collection 'contextrag' does not exist; refusing to create one", file=sys.stderr)
                return 2
            result["qdrant"] = QdrantVectorStore(1, client=client, cipher=cipher).rekey(dry_run=args.dry_run)
    except Exception as exc:  # nothing is half-applied: each record is verified before it is replaced
        print(f"rotate-key: stopped with {type(exc).__name__}; no record was left unreadable. Fix the cause and re-run.",
              file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


def _run_bench(args) -> int:
    from .eval import retrieval_bench as B

    if args.repeats < 1:
        print("bench: --repeats must be >= 1", file=sys.stderr)
        return 2
    try:
        report = B.run_benchmark(dataset=args.dataset or B.DATA_PATH, repeats=args.repeats)
    except (OSError, KeyError, ValueError) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"dataset v{report['dataset']['version']} sha256={report['dataset']['sha256'][:12]} "
          f"({report['dataset']['documents']} docs, {report['dataset']['queries']} queries)  [{report['caveat']}]")
    print(f"{'strategy':14s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MRR':>6s} {'nDCG@5':>7s} {'p50 ms':>8s} {'p95 ms':>8s}")
    for name, r in report["retrieval"].items():
        m = r["metrics"]["all"]
        print(f"{name:14s} {m['recall@1']:6.3f} {m['recall@5']:6.3f} {m['recall@10']:6.3f} {m['mrr']:6.3f} "
              f"{m['ndcg@5']:7.3f} {r['latency']['p50_ms']:8.3f} {r['latency']['p95_ms']:8.3f}")
    print("grounding:", json.dumps(report["grounding"]["answerable"]), json.dumps(report["grounding"]["unanswerable"]))
    if args.check_baseline:
        try:
            baseline = json.loads(B.BASELINE_PATH.read_text(encoding="utf-8"))
            problems = B.regressions(report, baseline)
        except (OSError, KeyError, ValueError) as exc:
            print(f"bench: cannot check the baseline: {exc}", file=sys.stderr)
            return 2
        for p in problems:
            print(f"REGRESSION {p}", file=sys.stderr)
        return 1 if problems else 0
    return 0


def _run_ragas(args) -> int:
    from .eval import ragas_eval as R

    try:
        settings = R.JudgeSettings.from_env()
        data = json.loads(Path(args.cases).read_text(encoding="utf-8"))
        from . import factory

        rag = factory.from_env()
        for d in data["documents"]:
            rag.ingestor.ingest_text(d["text"], tenant=args.tenant, doc_id=d["doc_id"], source=d["doc_id"])
        samples, reference_source = R.samples_from_rag(rag, data["cases"], tenant=args.tenant)
        metrics = tuple(m for m in args.metrics.split(",") if m)
        scorer = R.RagasScorer(settings)
    except (R.JudgeConfigError, OSError, KeyError, ValueError) as exc:
        print(f"ragas: {exc}", file=sys.stderr)
        return 2
    pipeline = {"parent_tokens": rag.cfg.parent_tokens, "child_tokens": rag.cfg.child_tokens,
                "top_k": rag.cfg.top_k, "context_budget_tokens": rag.cfg.context_budget_tokens}
    report = R.run(samples, scorer, settings, dataset=args.cases, pipeline=pipeline,
                   reference_source=reference_source, metrics=metrics)
    R.write_report(report, args.out)
    print(json.dumps({m: {"mean": s.mean, "scored": s.scored, "failed": s.failed, "not_run": s.not_run}
                      for m, s in report.metrics.items()}, indent=2))
    return 0 if report.ok() else 1


def _run_calibrate(args) -> int:
    from .eval import calibrate as C

    try:
        if bool(args.cases) == bool(args.synthetic):
            raise C.CalibrationError("give exactly one of --cases or --synthetic")
        if args.synthetic:
            from .eval.long_corpus import build_long_corpus

            data, kind, path = build_long_corpus(seed=args.seed), "synthetic", None
        else:
            bundled = Path(args.cases).resolve().is_relative_to(Path(__file__).parent / "eval" / "data")
            kind = "fixture" if bundled else "supplied"
            data, path = json.loads(Path(args.cases).read_text(encoding="utf-8")), args.cases
        verifier = json.loads(Path(args.verifier_cases).read_text(encoding="utf-8")) if args.verifier_cases else None
        report = C.calibrate(data, verifier_cases=verifier, data_kind=kind, data_path=path, seed=args.seed,
                             target_precision=args.target_precision)
    except (C.CalibrationError, OSError, KeyError, ValueError) as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        return 2
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.proposal:
        C.write_proposal(report, args.proposal)
    print(json.dumps({"chunk_grid": report["chunk_grid"]["recommended"],
                      "rerank_min_score": report["rerank_min_score"]["status"],
                      "nli_threshold": report["nli_threshold"]["status"]}, indent=2))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
