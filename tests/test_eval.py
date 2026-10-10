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
