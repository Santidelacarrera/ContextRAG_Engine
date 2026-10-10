"""``python -m contextrag {serve,ingest,ask,eval}``. Backends come from environment variables
(see ``contextrag.factory.from_env``); with none set everything runs in memory."""
from __future__ import annotations

import argparse
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

    e = sub.add_parser("eval", help="run the built-in (or custom) evaluation; exit 1 if anything is unsafe")
    e.add_argument("--verifier-cases", help="JSON file of {premise, claim, label, category, id}")
    e.add_argument("--rag-cases", help="JSON file of {documents, cases}")
    e.add_argument("--sweep", action="store_true", help="also print false-support/recall for a range of thresholds")
    e.add_argument("--json", action="store_true", help="print machine-readable results")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.cmd == "eval" else logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    csv = lambda v: tuple(x for x in v.split(",") if x)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("contextrag.api.server:app_factory", factory=True, host=args.host, port=args.port)
        return 0

    if args.cmd == "keygen":
        import os

        from .api.auth import KeyStore, new_entry

        key, digest, entry = new_entry(args.id, args.tenant, args.role, csv(args.groups))
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


if __name__ == "__main__":
    raise SystemExit(main())
