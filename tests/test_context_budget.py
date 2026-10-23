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


def test_a_parent_that_does_not_fit_falls_back_to_its_exact_child_span():
    from contextrag.context.budget import span_context
    from contextrag.core.models import ChildChunk

    text = words(40, "relleno") + " la auditoría se conserva 7 años " + words(40, "mas")
    parent = ParentChunk(id="P", doc_id="d", tenant="t", source="d.md", text=text, start=100, end=100 + len(text))
    start = 100 + text.index("la auditoría")
    end = 100 + text.index("7 años") + len("7 años")
    child = ChildChunk(id="C", parent_id="P", doc_id="d", tenant="t", text=text[start - 100:end - 100],
                       start=start, end=end)
    parent_ctx = Context(1, parent, 0.9)
    span = span_context(parent_ctx, child)
    out = assemble_evidence([parent_ctx], ContextBudget(20), spans={"P": span})
    assert len(out) == 1 and out[0].parent.id == "C"
    assert out[0].parent.text == child.text
    assert out[0].parent.start == start and out[0].parent.end == end  # absolute offsets: attribution stays exact


def test_pipeline_answers_from_a_span_when_the_parent_exceeds_the_budget():
    from contextrag import factory
    from contextrag.core.config import Settings
    from contextrag.core.models import Status

    filler = " ".join(f"párrafo{i} relleno sin relación alguna" for i in range(400))
    doc = f"# Retención de datos\n\n{filler}\n\nLa auditoría se conserva durante 7 años en almacenamiento frío.\n\n{filler}"
    rag = factory.offline(settings=Settings(parent_tokens=1024, child_tokens=128, child_overlap_tokens=16,
                                            context_budget_tokens=300, context_safety_margin=0.0))
    rag.ingestor.ingest_text(doc, tenant="acme", doc_id="long.md")
    a = rag.ask("¿Cuántos años se conserva la auditoría?", tenant="acme")
    assert a.status is Status.VERIFIED and "7 años" in a.text
    ev = a.claims[0].evidence
    assert doc[ev.start:ev.end] == ev.text  # the claim's coordinates point into the original document
