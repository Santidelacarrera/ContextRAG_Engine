from contextrag import factory
from contextrag.eval.harness import eval_rag, eval_verifier
from contextrag.verification.nli_verifier import LexicalEntailment


def test_lexical_nli_never_supports_an_adversarial_claim():
    r = eval_verifier(LexicalEntailment())
    assert r.false_supports == [] and r.false_support_rate == 0.0
    assert r.recall == 1.0 and r.missed_supports == []
    assert r.by_category["role_swap"] == 1.0 and r.total >= 20


def test_a_bag_of_words_model_is_caught_by_the_harness():
    class BagOfWords:  # the previous, order-blind behaviour
        def entailment(self, pairs):
            import re

            w = lambda t: set(re.findall(r"\w+", t.lower()))
            return [len(w(h) & w(p)) / len(w(h)) for p, h in pairs]

    r = eval_verifier(BagOfWords())
    assert "bad-swap-es" in r.false_supports and r.false_support_rate > 0


def test_end_to_end_eval_is_safe_and_complete():
    r = eval_rag(factory.offline)
    assert r.unsafe == [] and r.unsafe_rate == 0.0
    assert r.answer_rate == r.correct_rate == r.retrieval_hit_rate == 1.0
    assert r.abstention_rate == 1.0 and r.mrr >= 0.9 and r.failures == []


def test_eval_flags_an_unsafe_system():
    class Gullible:
        def generate(self, question, contexts, feedback=None):
            return "El mundial de 1998 lo ganó Francia. [1]"

    class AcceptAll:
        def entailment(self, pairs):
            return [1.0] * len(pairs)

    r = eval_rag(lambda: _rag(Gullible(), AcceptAll()))
    assert r.unsafe_rate > 0 and r.correct_rate < r.answer_rate and "a1" in r.unsafe


def _rag(llm, nli):
    rag = factory.offline(llm=llm)
    rag.verifier.nli = nli
    rag.reranker = type("NoFloor", (), {"score": lambda self, q, ps: [1.0] * len(ps)})()  # disable relevance gate
    return rag


def test_threshold_sweep_recommends_lowest_safe_threshold():
    from contextrag.eval.harness import sweep_thresholds

    class Graded:  # a "semantic" model whose scores overlap: false claims reach up to 0.93, true ones from 0.9
        table = {"bad": 0.93, "ok": 0.97}

        def entailment(self, pairs):
            return [self.table["ok" if p.endswith("(ok)") else "bad"] for p, _ in pairs]

    cases = [{"id": "1", "label": "supported", "category": "c", "premise": "x (ok)", "claim": "x."},
             {"id": "2", "label": "unsupported", "category": "c", "premise": "y (bad)", "claim": "y."}]
    out = sweep_thresholds(Graded(), cases, thresholds=(0.9, 0.95, 0.99))
    rows = {r["threshold"]: r for r in out["rows"]}
    assert rows[0.9]["false_support_rate"] == 1.0 and rows[0.95]["false_support_rate"] == 0.0
    assert rows[0.95]["recall"] == 1.0 and rows[0.99]["recall"] == 0.0
    assert out["recommended"] == 0.95
    assert sweep_thresholds(type("A", (), {"entailment": lambda s, p: [1.0] * len(p)})(), cases)["recommended"] is None


def test_cli_eval_sweep(capsys):
    import json

    from contextrag.__main__ import main

    assert main(["eval", "--sweep", "--json"]) == 0
    sweep = json.loads(capsys.readouterr().out)["sweep"]
    rows = {r["threshold"]: r for r in sweep["rows"]}
    assert rows[0.5]["false_support_rate"] > 0  # permissive thresholds let partially-supported claims through
    assert rows[0.95]["false_support_rate"] == 0 and rows[0.95]["recall"] == 1.0
    rec = sweep["recommended"]
    assert rec is not None and rec <= 0.95 and rows[rec]["false_support_rate"] == 0
    assert all(rows[t]["false_support_rate"] > 0 for t in rows if t < rec)  # and it is the lowest such threshold
    assert main(["eval", "--sweep"]) == 0 and "recommended threshold" in capsys.readouterr().out
