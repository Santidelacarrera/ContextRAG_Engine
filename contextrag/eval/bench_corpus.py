"""Deterministic synthetic benchmark corpus (version 1).

Forty fictional service runbooks and seventy questions, all generated from a seed, so the checked-in dataset can be
verified by regenerating it (``python -m contextrag.eval.bench_corpus --check``). Nothing here is real or personal data.

Question kinds and what each is for:

* ``direct`` (30): the question uses the document's own wording. The easy case for lexical retrieval.
* ``paraphrase`` (15): the same fact asked with different words; only the service name overlaps.
* ``multi`` (15): "which services ..." questions with several relevant documents, so ranking quality (MRR, nDCG) is
  meaningful and Recall@k can be below 1 for a good retriever.
* ``unanswerable`` (10): the service does not exist, or the topic is off-corpus. The system should abstain.

Relevance is judged at document level. ``expect`` (direct and paraphrase only) is the literal value the answer must
contain to count as correct.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

VERSION = 1
SEED = 11
N_DOCS = 40
DATA_PATH = Path(__file__).parent / "data" / "retrieval_bench_v1.json"

_SYLLABLES = ["al", "ber", "cor", "dun", "el", "fa", "gor", "hal", "is", "jun", "kel", "lor", "mar", "nor", "ov",
              "pel", "quin", "ros", "sal", "tor", "ul", "ver", "wen", "xan", "yor", "zel"]
RETENTION = [30, 60, 90, 180, 365, 730, 2555]
REGIONS = ["eu-west", "eu-north", "us-east", "us-west", "ap-south", "ap-east", "sa-east", "me-central", "af-south",
           "ca-central"]
TEAMS = ["Payments", "Identity", "Search", "Billing", "Platform", "Analytics", "Messaging", "Storage", "Gateway",
         "Compliance"]
CIPHERS = ["AES-256", "AES-128", "ChaCha20", "RSA-4096"]
ONCALL = ["around the clock", "during business hours only", "by a follow-the-sun rotation"]

_FILLER = [
    "Changes to this service follow the standard review process and are announced in the engineering channel.",
    "Dashboards for latency and error budgets are linked from the internal portal.",
    "Runbook steps are reviewed every quarter and updated after each major incident.",
    "Capacity is planned against the traffic forecast published at the start of the year.",
    "Access to production is granted per task and revoked automatically afterwards.",
    "Dependencies are listed in the architecture register together with their owners.",
    "Backups are verified by restoring them into an isolated environment.",
    "Feature flags are cleaned up within two sprints of a full rollout.",
]


def _name(rng: random.Random, used: set[str]) -> str:
    while True:
        n = "".join(rng.choice(_SYLLABLES) for _ in range(rng.choice((2, 3)))).capitalize() + "x"
        if n not in used:
            used.add(n)
            return n


def _document(name: str, a: dict, rng: random.Random) -> str:
    filler = rng.sample(_FILLER, 3)
    return (
        f"# {name} service runbook\n\n"
        f"{name} is an internal service owned by the {a['team']} team.\n\n"
        f"## Data retention\n\nThe retention period for {name} logs is {a['retention']} days. "
        f"After that period the records are deleted automatically.\n\n"
        f"## Hosting\n\n{name} runs in the {a['region']} region. {filler[0]}\n\n"
        f"## Security\n\n{name} encrypts data at rest using {a['cipher']}. Keys are rotated every "
        f"{a['rotation']} months. {filler[1]}\n\n"
        f"## Operations\n\nSupport for {name} is provided {a['oncall']}. {filler[2]}\n"
    )


def build_bench(seed: int = SEED) -> dict:
    # seeded on purpose (reproducible evaluation data, not security-sensitive)
    rng = random.Random(seed)  # nosec B311
    used: set[str] = set()
    services = []
    for _ in range(N_DOCS):
        name = _name(rng, used)
        attrs = {"team": rng.choice(TEAMS), "retention": rng.choice(RETENTION), "region": rng.choice(REGIONS),
                 "cipher": rng.choice(CIPHERS), "rotation": rng.choice((3, 6, 12)), "oncall": rng.choice(ONCALL)}
        services.append((name, attrs))
    documents = [{"doc_id": f"{name.lower()}.md", "text": _document(name, a, rng)} for name, a in services]

    direct = {"retention": "What is the retention period for {n} logs?", "region": "In which region does {n} run?",
              "cipher": "Which algorithm does {n} use to encrypt data at rest?", "team": "Which team owns {n}?",
              "oncall": "How is support for {n} provided?"}
    paraphrase = {"retention": "For how long does {n} hold on to its log records?", "region": "Where is {n} hosted?",
                  "cipher": "What cipher protects {n} stored data?", "team": "Who is responsible for {n}?",
                  "oncall": "When can someone be reached for {n} incidents?"}
    expected = {"retention": lambda a: f"{a['retention']} days", "region": lambda a: a["region"],
                "cipher": lambda a: a["cipher"], "team": lambda a: a["team"], "oncall": lambda a: a["oncall"]}
    attrs_cycle = list(direct)

    queries: list[dict] = []
    order = list(range(N_DOCS))
    rng.shuffle(order)
    for i in range(30):  # direct: each service at most once, attributes cycled so all five are exercised
        name, a = services[order[i]]
        attr = attrs_cycle[i % 5]
        queries.append({"id": f"d{i + 1:02d}", "kind": "direct", "question": direct[attr].format(n=name),
                        "relevant": [f"{name.lower()}.md"], "expect": expected[attr](a)})
    for i in range(15):  # paraphrase: services the direct set did not use
        name, a = services[order[30 + i % 10]]
        attr = attrs_cycle[(i + 2) % 5]
        queries.append({"id": f"p{i + 1:02d}", "kind": "paraphrase", "question": paraphrase[attr].format(n=name),
                        "relevant": [f"{name.lower()}.md"], "expect": expected[attr](a)})

    multi: list[dict] = []
    for attr, values, template in (("region", REGIONS, "Which services run in the {v} region?"),
                                   ("team", TEAMS, "Which services are owned by the {v} team?")):
        for v in values:
            rel = [f"{n.lower()}.md" for n, a in services if a[attr] == v]
            if 2 <= len(rel) <= 8:
                multi.append({"kind": "multi", "question": template.format(v=v), "relevant": rel})
    for i, q in enumerate(multi[:15]):
        queries.append({"id": f"m{i + 1:02d}", **q})

    for i in range(5):  # a service that does not exist
        queries.append({"id": f"u{i + 1:02d}", "kind": "unanswerable", "relevant": [],
                        "question": f"What is the retention period for {_name(rng, used)} logs?"})
    off_topic = ["What is the capital of Atlantis?", "Who won the regional chess tournament last year?",
                 "How many moons does the planet Zorbon have?", "What is the recipe for the cafeteria soup?",
                 "Which airline serves the new northern airport?"]
    for i, text in enumerate(off_topic):
        queries.append({"id": f"u{i + 6:02d}", "kind": "unanswerable", "relevant": [], "question": text})

    return {"version": VERSION, "seed": seed, "synthetic": True, "documents": documents, "queries": queries}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="write the dataset file")
    ap.add_argument("--check", action="store_true", help="exit 1 if the checked-in file differs from a regeneration")
    args = ap.parse_args(argv)
    text = json.dumps(build_bench(), indent=1, ensure_ascii=False) + "\n"
    if args.write:
        DATA_PATH.write_text(text, encoding="utf-8")
        return 0
    if args.check:
        same = DATA_PATH.exists() and DATA_PATH.read_text(encoding="utf-8") == text
        if not same:
            print("retrieval_bench_v1.json differs from build_bench(); regenerate with --write", file=sys.stderr)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
