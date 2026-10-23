"""Evidence assembly: budget never exceeded, dedupe, deterministic edge placement, citations renumbered."""
import pytest

from contextrag.context.budget import ContextBudget, assemble_evidence, edge_order, jaccard
from contextrag.core.chunking import count_tokens
from contextrag.core.models import Context, ParentChunk


def ctx(pid, text, score):
    p = ParentChunk(id=pid, doc_id=f"doc-{pid}", tenant="t", source=f"{pid}.md", text=text, start=0, end=len(text))
    return Context(index=0, parent=p, score=score)


def words(n, tag):
    return " ".join(f"{tag}{i}" for i in range(n))


def test_budget_is_never_exceeded_and_small_parents_fill_the_gap():
    cands = [ctx("a", words(60, "a"), 0.9), ctx("b", words(60, "b"), 0.8),
             ctx("c", words(20, "c"), 0.7)]  # 'b' does not fit after 'a' with a 90-token budget; 'c' does
    budget = ContextBudget(evidence_tokens=100, safety_margin=0.1)  # effective 90
    out = assemble_evidence(cands, budget)
    assert sum(count_tokens(c.parent.text) for c in out) <= budget.effective_tokens
    assert {c.parent.id for c in out} == {"a", "c"}


def test_near_duplicate_parents_are_dropped_keeping_the_better_ranked_one():
    original = words(40, "shared")
    copy = original + " x"  # a near-identical passage from another document version
    out = assemble_evidence([ctx("best", original, 0.9), ctx("dup", copy, 0.8), ctx("other", words(30, "o"), 0.5)],
                            ContextBudget(10_000), dedupe_threshold=0.8)
    ids = {c.parent.id for c in out}
    assert "best" in ids and "dup" not in ids and "other" in ids


def test_edge_placement_puts_best_first_second_best_last_and_weakest_in_the_middle():
    ranked = [ctx(str(i), words(3, f"p{i}"), 1.0 - i / 10) for i in range(1, 6)]  # ranks 1..5
    assert [c.parent.id for c in edge_order(ranked)] == ["1", "3", "5", "4", "2"]
    out = assemble_evidence(ranked, ContextBudget(10_000), placement="edges")
    assert [c.parent.id for c in out] == ["1", "3", "5", "4", "2"]
    assert [c.index for c in out] == [1, 2, 3, 4, 5]  # citations follow prompt position


def test_descending_placement_keeps_rank_order_and_is_deterministic():
    cands = [ctx("x", words(5, "x"), 0.2), ctx("y", words(5, "y"), 0.9), ctx("z", words(5, "z"), 0.5)]
    first = assemble_evidence(cands, ContextBudget(10_000), placement="descending")
    second = assemble_evidence(list(reversed(cands)), ContextBudget(10_000), placement="descending")
    assert [c.parent.id for c in first] == ["y", "z", "x"]
    assert [(c.parent.id, c.index) for c in first] == [(c.parent.id, c.index) for c in second]


def test_a_parent_larger_than_the_whole_budget_is_skipped_not_truncated():
    out = assemble_evidence([ctx("huge", words(500, "h"), 0.9), ctx("small", words(5, "s"), 0.1)],
                            ContextBudget(50))
    assert [c.parent.id for c in out] == ["small"]
    assert out[0].parent.text == words(5, "s")  # the passage is whole, never cut mid-sentence


def test_empty_input_and_invalid_configuration():
    assert assemble_evidence([], ContextBudget(100)) == []
    with pytest.raises(ValueError):
        ContextBudget(0)
    with pytest.raises(ValueError):
        ContextBudget(100, safety_margin=1.0)
    with pytest.raises(ValueError):
        assemble_evidence([], ContextBudget(100), placement="random")
    with pytest.raises(ValueError):
        assemble_evidence([], ContextBudget(100), dedupe_threshold=0)


def test_jaccard_edges():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_pipeline_answers_within_the_budget_and_falls_back_when_nothing_fits():
    from contextrag import factory
    from contextrag.core.config import Settings
    from contextrag.core.models import Status
    from conftest import POLICY

    rag = factory.offline(settings=Settings(context_budget_tokens=5, context_safety_margin=0.0))
    rag.ingestor.ingest_text(POLICY, tenant="acme", doc_id="policy.md")
    a = rag.ask("¿Cuánto tiempo se conservan los registros de auditoría?", tenant="acme")
    assert a.status is Status.FALLBACK and "budget" in a.reason
    assert a.sources == []  # nothing that did not fit was sent to the model
