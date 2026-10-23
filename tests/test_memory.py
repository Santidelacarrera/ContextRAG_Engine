"""Conversational memory: versioned turns, idempotent and restart-safe compaction, bounded scheduling."""
import pytest

from contextrag.memory.store import (ConversationNotFound, ConversationStore, MemoryDraft, QueueFull,
                                     VersionConflict)
from contextrag.memory.summarizer import ExtractiveSummarizer
from contextrag.memory.worker import CompactionWorker


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def store(path=":memory:", clock=None, **kw):
    return ConversationStore(path, clock=clock or Clock(), **kw)


def fill(s, cid="c1", tenant="acme", n=6, text="texto de prueba normal"):
    s.create_conversation(tenant, cid)
    for i in range(n):
        s.append_turn(tenant, cid, "user" if i % 2 == 0 else "assistant", f"{text} {i}")
    return cid


# ---- versioned turns ------------------------------------------------------------------------------

def test_turns_are_contiguous_and_each_append_bumps_the_version():
    s = store()
    cid = fill(s, n=4)
    assert s.version("acme", cid) == 4
    assert [t.seq for t in s.turns_between("acme", cid, 0, 4)] == [1, 2, 3, 4]


def test_optimistic_concurrency_refuses_a_stale_writer_and_writes_nothing():
    s = store()
    cid = fill(s, n=2)
    s.append_turn("acme", cid, "user", "otro mensaje", expected_version=2)
    with pytest.raises(VersionConflict) as info:
        s.append_turn("acme", cid, "user", "escrito con versión vieja", expected_version=2)
    assert info.value.expected == 2 and info.value.actual == 3
    assert s.version("acme", cid) == 3 and len(s.turns_between("acme", cid, 0, 99)) == 3


@pytest.mark.parametrize("role,content", [("system", "x"), ("user", "   "), ("user", "x" * 20_001)])
def test_invalid_turns_are_rejected(role, content):
    s = store()
    cid = fill(s, n=0)
    with pytest.raises(ValueError):
        s.append_turn("acme", cid, role, content)


def test_unknown_conversation_is_not_created_implicitly():
    with pytest.raises(ConversationNotFound):
        store().append_turn("acme", "nope", "user", "hola")


def test_history_window_keeps_the_newest_contiguous_turns_within_budget():
    s = store()
    cid = fill(s, n=0)
    for words in (["uno"] * 3, ["dos"] * 3, ["tres"] * 3):
        s.append_turn("acme", cid, "user", " ".join(words))
    window = s.history_window("acme", cid, token_budget=6)  # each turn costs 3 tokens: two fit
    assert [t.content.split()[0] for t in window] == ["dos", "tres"]


# ---- compaction: planning, commit, provenance ----------------------------------------------------

def test_planning_is_idempotent_and_skips_histories_under_threshold():
    s = store()
    cid = fill(s, n=8)
    assert s.plan_compaction("acme", cid, threshold_tokens=10**6, keep_recent_turns=2) is None
    first = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=2)
    again = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=2)
    assert first and first == again and len(s.active_jobs()) == 1


def test_a_full_cycle_stores_only_user_statements_and_keeps_all_source_turns():
    s = store()
    cid = s.create_conversation("acme", "c1") or "c1"
    s.append_turn("acme", cid, "user", "Decidimos usar Postgres para la auditoría.")
    s.append_turn("acme", cid, "assistant", "Perfecto, lo documento como decisión del modelo.")
    s.append_turn("acme", cid, "user", "Prefiero respuestas breves.")
    s.append_turn("acme", cid, "user", "Pendiente: revisar el índice de particiones.")
    s.append_turn("acme", cid, "user", "Hoy hace buen tiempo.")
    s.append_turn("acme", cid, "assistant", "Entendido.")
    s.append_turn("acme", cid, "user", "Reciente sin resumir.")
    w = CompactionWorker(s, ExtractiveSummarizer(), workers=1)
    try:
        assert w.schedule("acme", cid, threshold_tokens=0, keep_recent_turns=2)
        assert w.drain(timeout=10)
    finally:
        w.close()
    mem = s.memories("acme", cid)
    assert {m.memory_type for m in mem} == {"decision", "preference", "task"}
    assert all(m.origin == "user_stated" and m.confidence == 1.0 for m in mem)
    assert not any("buen tiempo" in m.content or "Entendido" in m.content for m in mem)  # chatter and bot text
    assert len(s.turns_between("acme", cid, 0, 99)) == 7  # provenance: nothing deleted
    assert s.uncompacted_tokens("acme", cid) > 0  # the two most recent turns stay uncompacted


def test_a_turn_arriving_during_compaction_is_preserved_and_not_summarized_away():
    s = store()
    cid = fill(s, n=6)
    claim_job = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=1)
    claim = s.claim_job(claim_job, lease_seconds=60)
    s.append_turn("acme", cid, "user", "Mensaje nuevo llegado durante el resumen.")  # after the snapshot
    assert s.complete_job(claim, [MemoryDraft("task", "Tarea antigua", "user_stated", 1.0, 2)])
    assert [t.content for t in s.turns_between("acme", cid, 6, 7)] == ["Mensaje nuevo llegado durante el resumen."]
    assert s.version("acme", cid) == 7


def test_a_stale_job_cannot_regress_newer_compaction_or_duplicate_memories():
    s = store()
    cid = fill(s, n=10)
    older = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=6)  # through 4
    newer = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=2)  # through 8
    assert older != newer
    c_new = s.claim_job(newer, lease_seconds=60)
    assert s.complete_job(c_new, [MemoryDraft("decision", "Decisión nueva", "user_stated", 1.0, 3)])
    c_old = s.claim_job(older, lease_seconds=60)
    assert c_old is not None
    assert s.complete_job(c_old, [MemoryDraft("decision", "Decisión vieja", "user_stated", 1.0, 2)]) is False
    assert [m.content for m in s.memories("acme", cid)] == ["Decisión nueva"]


def test_only_the_claim_holder_can_commit_after_a_lease_takeover():
    clock = Clock()
    s = store(clock=clock, max_attempts=3)
    cid = fill(s, n=6)
    job = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=1)
    first = s.claim_job(job, lease_seconds=60)
    clock.t += 120  # the first worker died: its lease expired
    second = s.claim_job(job, lease_seconds=60)
    assert second is not None and second.attempt == first.attempt + 1
    draft = [MemoryDraft("task", "Tarea única", "user_stated", 1.0, 2)]
    assert s.complete_job(first, draft) is False  # the zombie cannot commit
    assert s.complete_job(second, draft) is True
    assert len(s.memories("acme", cid)) == 1


def test_draft_outside_the_job_range_is_rejected_and_the_job_can_be_failed():
    s = store()
    cid = fill(s, n=6)
    job = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=1)
    claim = s.claim_job(job, lease_seconds=60)
    with pytest.raises(ValueError, match="outside the job range"):
        s.complete_job(claim, [MemoryDraft("task", "fuera", "user_stated", 1.0, 6)])
    assert s.memories("acme", cid) == []
    assert s.fail_job(claim, "ValueError") == "queued"


# ---- failures, retries and bounded resources --------------------------------------------------------

class Boom:
    calls = 0

    def summarize(self, turns):
        Boom.calls += 1
        raise RuntimeError("provider down: sensitive text " + turns[0].content)


def test_failures_are_retried_then_marked_failed_with_a_code_and_no_content():
    s = store(max_attempts=2)
    cid = fill(s, n=6)
    w = CompactionWorker(s, Boom(), workers=1)
    try:
        job = w.schedule("acme", cid, threshold_tokens=0, keep_recent_turns=1)
        assert w.drain(timeout=10)
    finally:
        w.close()
    assert s.job_state(job) == "failed" and Boom.calls == 2
    assert s.memories("acme", cid) == []
    row = s._db.execute("SELECT error FROM compaction_jobs WHERE job_id = ?", (job,)).fetchone()
    assert row[0] == "RuntimeError" and "sensitive" not in row[0]


def test_the_queue_is_bounded_and_scheduling_is_refused_beyond_the_limit():
    s = store(queue_limit=2)
    for i in range(3):
        cid = f"c{i}"
        fill(s, cid=cid, n=4)
    s.plan_compaction("acme", "c0", threshold_tokens=0, keep_recent_turns=1)
    s.plan_compaction("acme", "c1", threshold_tokens=0, keep_recent_turns=1)
    with pytest.raises(QueueFull):
        s.plan_compaction("acme", "c2", threshold_tokens=0, keep_recent_turns=1)


def test_closing_the_worker_leaves_queued_work_recoverable_not_lost():
    s = store()
    cid = fill(s, n=6)
    w = CompactionWorker(s, ExtractiveSummarizer(), workers=1)
    w.close()
    with pytest.raises(RuntimeError):
        w.schedule("acme", cid, threshold_tokens=0, keep_recent_turns=1)
    assert s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=1) in s.active_jobs()


# ---- durability, tenancy, erasure -----------------------------------------------------------------

def test_memories_and_jobs_survive_a_restart_and_a_crashed_run_is_recovered(tmp_path):
    path = tmp_path / "memory.db"
    clock = Clock()
    s = store(path, clock=clock)
    cid = fill(s, n=6)
    job = s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=1)
    s.claim_job(job, lease_seconds=60)  # the process dies while holding the claim
    s.close()

    clock.t += 3600  # restart much later
    s2 = store(path, clock=clock)
    w = CompactionWorker(s2, ExtractiveSummarizer(), workers=1, lease_seconds=60)
    try:
        assert w.recover() == [job]
        assert w.drain(timeout=10)
    finally:
        w.close()
    assert s2.job_state(job) == "done"
    assert s2.version("acme", cid) == 6 and s2.turns_between("acme", cid, 0, 99)


def test_tenants_never_see_each_others_memories_or_turns():
    s = store()
    fill(s, cid="shared", tenant="acme", n=4)
    fill(s, cid="shared", tenant="rival", n=4)
    s.append_turn("acme", "shared", "user", "Decidimos algo secreto de acme.")
    assert s.memories("rival", "shared") == []
    with pytest.raises(ConversationNotFound):
        s.version("otro", "shared")


def test_relevant_memories_selects_by_query_and_is_bounded():
    s = store()
    cid = fill(s, n=0)
    for line in ("Decidimos usar Qdrant para vectores.", "Pendiente: migrar el backup de Redis."):
        s.append_turn("acme", cid, "user", line)
    w = CompactionWorker(s, ExtractiveSummarizer(), workers=1)
    try:
        w.schedule("acme", cid, threshold_tokens=0, keep_recent_turns=0)
        assert w.drain(timeout=10)
    finally:
        w.close()

    def overlap(q, text):
        return len(set(q.lower().split()) & set(text.lower().split()))

    hits = s.relevant_memories("acme", cid, "¿qué usamos para vectores en Qdrant?", k=1, score=overlap)
    assert [m.content for m in hits] == ["Decidimos usar Qdrant para vectores."]
    assert s.relevant_memories("acme", cid, "clima", k=5, score=overlap) == []


def test_erasure_removes_the_conversation_everywhere():
    s = store()
    cid = fill(s, n=6)
    s.plan_compaction("acme", cid, threshold_tokens=0, keep_recent_turns=1)
    s.delete_conversation("acme", cid)
    with pytest.raises(ConversationNotFound):
        s.version("acme", cid)
    assert s.active_jobs() == []
