"""Authorisation is enforced before generation: a passage the caller may not read never reaches the model.

The checks look at what the model *receives*, not at the final answer, because a filter applied only to the output would
leave the data in the prompt, in logs and in the provider's hands.
"""
import pytest

from contextrag import factory
from contextrag.eval.entity_bench import TENANT, CONFIGS
from contextrag.eval.entity_corpus import build_entity_bench
from contextrag.generation.llm import ExtractiveLLM


class Spy(ExtractiveLLM):
    """Records every passage handed to the model, then answers exactly like the extractive generator."""

    def __init__(self):
        super().__init__()
        self.seen: list[set[str]] = []

    def generate(self, question, contexts, feedback=None, *, memory=None):
        self.seen.append({c.parent.doc_id for c in contexts} | {c.parent.text for c in contexts})
        return super().generate(question, contexts, feedback, memory=memory)


@pytest.fixture(scope="module")
def world():
    data = build_entity_bench()
    spy = Spy()
    rag = factory.offline(llm=spy)
    for d in data["documents"]:
        rag.ingestor.ingest_text(d["text"], tenant=TENANT, doc_id=d["doc_id"], source=d["doc_id"], acl=tuple(d["acl"]))
    yield rag, data, spy
    rag.close()


def restricted_docs(data):
    return {d["doc_id"]: d["text"] for d in data["documents"] if d["acl"]}


def test_no_restricted_passage_reaches_the_model_for_a_caller_without_the_group(world):
    rag, data, spy = world
    secrets = restricted_docs(data)
    assert secrets, "the synthetic corpus must contain restricted documents for this test to mean anything"
    for q in data["queries"]:
        spy.seen.clear()
        rag.ask(q["question"], tenant=TENANT, groups=frozenset())  # no group at all
        for seen in spy.seen:
            for doc_id, text in secrets.items():
                assert doc_id not in seen and text not in seen, (q["id"], doc_id)


def test_the_restricted_passage_does_reach_a_caller_who_holds_the_group(world):
    """The control is not simply refusing everything: the authorised caller does get the document."""
    rag, data, spy = world
    spy.seen.clear()
    a = rag.ask("¿Dónde tiene su domicilio SR?", tenant=TENANT, groups=frozenset({"caso-x"}))
    assert a.status.value == "verified"
    assert any("reservado.md" in seen for seen in spy.seen)


def test_an_alias_lookup_cannot_fetch_a_restricted_declaration(world):
    """Alias resolution reads declarations through the same ACL-filtered retrieval, so a restricted alias stays hidden."""
    rag, data, _ = world
    a = rag.ask("¿Cuál es el capital social de SR?", tenant=TENANT, groups=frozenset())
    text = str(a.to_response())
    assert "Osorno" not in text and "Reservada" not in text and "reservado.md" not in text


def test_the_entity_benchmark_configurations_all_respect_the_same_boundary():
    assert set(CONFIGS) == {"no_gate", "gate_only", "alias_resolution"}
