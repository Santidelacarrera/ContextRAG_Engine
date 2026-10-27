"""Encryption at rest for conversational memory: round trip, fail-closed reads, position binding, legacy migration."""
import sqlite3

import pytest

from contextrag.core.security import TenantCipher
from contextrag.memory.store import (ConversationStore, MemoryDraft, MemoryIntegrityError)
from contextrag.memory.summarizer import ExtractiveSummarizer
from contextrag.memory.worker import CompactionWorker

KEY = b"0123456789abcdef0123456789abcdef"
OTHER_KEY = b"fedcba9876543210fedcba9876543210"
SECRET = "codename ORCHID-7731 is the production database password"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def make(path, key=KEY, **kw):
    return ConversationStore(path, cipher=TenantCipher(key), clock=Clock(), **kw)


def raw_rows(path, table):
    con = sqlite3.connect(path)
    try:
        return con.execute(f"SELECT * FROM {table}").fetchall()
    finally:
        con.close()


def file_bytes(path):
    """Every byte the database and its write-ahead log hold on disk."""
    import pathlib

    data = b""
    for suffix in ("", "-wal", "-shm"):
        p = pathlib.Path(str(path) + suffix)
        if p.exists():
            data += p.read_bytes()
    return data


# ---- round trip and what is on disk --------------------------------------------------------------

def test_turns_round_trip_and_the_database_holds_no_plaintext(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.append_turn("acme", "c1", "user", SECRET)
    assert s.turns_between("acme", "c1", 0, 1)[0].content == SECRET
    s.close()
    assert SECRET.encode() not in file_bytes(path)
    con = sqlite3.connect(path)
    content, encrypted = con.execute("SELECT content, encrypted FROM turns").fetchone()
    con.close()
    assert content != SECRET and encrypted == 1  # content is ciphertext, encrypted flag is set


def test_memories_round_trip_and_are_stored_encrypted(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.append_turn("acme", "c1", "user", "we decided to use the blue cluster")
    s.append_turn("acme", "c1", "assistant", "noted")
    claim = s.claim_job(_plan(s, "acme", "c1"), lease_seconds=60)
    s.complete_job(claim, [MemoryDraft("decision", SECRET, "user_stated", 1.0, 1)])
    assert s.memories("acme", "c1")[0].content == SECRET
    s.close()
    assert SECRET.encode() not in file_bytes(path)


def _plan(s, tenant, cid):
    job = s.plan_compaction(tenant, cid, threshold_tokens=0, keep_recent_turns=1)
    assert job is not None
    return job


# ---- construction ----------------------------------------------------------------------------------

def test_a_store_without_a_cipher_is_refused_unless_plaintext_is_explicit():
    with pytest.raises(ValueError, match="requires a TenantCipher"):
        ConversationStore(":memory:")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConversationStore(":memory:", cipher=TenantCipher(KEY), plaintext=True)
    assert ConversationStore(":memory:", plaintext=True).encrypted is False


def test_the_cipher_key_must_be_long_enough():
    with pytest.raises(ValueError):
        TenantCipher(b"short")


# ---- fail closed ------------------------------------------------------------------------------------

def test_a_wrong_master_key_is_refused_not_decrypted(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.append_turn("acme", "c1", "user", SECRET)
    s.close()
    intruder = make(path, key=OTHER_KEY)
    with pytest.raises(MemoryIntegrityError, match="failed authentication"):
        intruder.turns_between("acme", "c1", 0, 1)


def test_another_tenants_key_cannot_read_a_tenants_record(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.create_conversation("globex", "c1")
    s.append_turn("acme", "c1", "user", SECRET)
    s.append_turn("globex", "c1", "user", "globex data")
    # Copy acme's ciphertext into globex's row: a store-level swap, not something the API can do.
    con = sqlite3.connect(path)
    acme_cipher = con.execute("SELECT content FROM turns WHERE tenant = 'acme'").fetchone()[0]
    con.execute("UPDATE turns SET content = ? WHERE tenant = 'globex'", (acme_cipher,))
    con.commit()
    con.close()
    with pytest.raises(MemoryIntegrityError):
        s.turns_between("globex", "c1", 0, 1)
    assert s.turns_between("acme", "c1", 0, 1)[0].content == SECRET  # the owner still reads its own


def test_a_row_moved_to_another_position_in_the_same_conversation_is_refused(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.append_turn("acme", "c1", "user", "first")
    s.append_turn("acme", "c1", "user", "second")
    con = sqlite3.connect(path)
    seq1, seq2 = (r[0] for r in con.execute("SELECT content FROM turns ORDER BY seq").fetchall())
    con.execute("UPDATE turns SET content = ? WHERE seq = 1", (seq2,))  # the same key, a different position
    con.commit()
    con.close()
    with pytest.raises(MemoryIntegrityError):
        s.turns_between("acme", "c1", 0, 2)
    assert seq1 != seq2


def test_tampered_ciphertext_is_refused(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.append_turn("acme", "c1", "user", SECRET)
    con = sqlite3.connect(path)
    token = con.execute("SELECT content FROM turns").fetchone()[0]
    flipped = token[:-2] + ("A" if token[-2] != "A" else "B") + token[-1]
    con.execute("UPDATE turns SET content = ?", (flipped,))
    con.commit()
    con.close()
    with pytest.raises(MemoryIntegrityError):
        s.turns_between("acme", "c1", 0, 1)


def test_malformed_stored_values_are_refused(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.append_turn("acme", "c1", "user", SECRET)
    con = sqlite3.connect(path)
    con.execute("UPDATE turns SET content = 'not-a-token'")
    con.commit()
    con.close()
    with pytest.raises(MemoryIntegrityError):
        s.history_window("acme", "c1", 10_000)


def test_an_empty_tenant_is_refused_and_never_uses_a_shared_key():
    s = make(":memory:")
    with pytest.raises(ValueError, match="tenant is required"):
        s._store_text("", "turn:c1:1", "x")


def test_the_error_message_names_a_position_not_the_content(tmp_path):
    path = tmp_path / "memory.db"
    s = make(path)
    s.create_conversation("acme", "c1")
    s.append_turn("acme", "c1", "user", SECRET)
    s2 = make(path, key=OTHER_KEY)
    with pytest.raises(MemoryIntegrityError) as info:
        s2.turns_between("acme", "c1", 0, 1)
    assert "turn:c1:1" in str(info.value)
    assert SECRET not in str(info.value)


# ---- compaction and retry paths -----------------------------------------------------------------------

def test_a_failing_attempt_records_only_an_error_code_and_the_retry_writes_encrypted_memory(tmp_path):
    path = tmp_path / "memory.db"
    store = make(path, max_attempts=3)
    store.create_conversation("acme", "c1")
    for i in range(4):
        store.append_turn("acme", "c1", "user" if i % 2 == 0 else "assistant", f"{SECRET} number {i}")

    class FlakySummarizer(ExtractiveSummarizer):
        calls = 0

        def summarize(self, turns):
            FlakySummarizer.calls += 1
            if FlakySummarizer.calls == 1:
                raise RuntimeError(f"model choked on {turns[0].content}")  # content must not reach the store
            return super().summarize(turns)

    worker = CompactionWorker(store, FlakySummarizer(), workers=1)
    try:
        job = worker.schedule("acme", "c1", threshold_tokens=0, keep_recent_turns=1)
        assert worker.drain(timeout=10)
    finally:
        worker.close()
    assert store.job_state(job) == "done"
    jobs = raw_rows(path, "compaction_jobs")
    assert all(SECRET not in str(row) for row in jobs)  # the failed attempt's error is the type name only
    assert SECRET.encode() not in file_bytes(path)


# ---- migration of legacy plaintext ----------------------------------------------------------------

LEGACY_SCHEMA = """
CREATE TABLE conversations (tenant TEXT NOT NULL, conversation_id TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 0,
    compacted_through INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, PRIMARY KEY (tenant, conversation_id));
CREATE TABLE turns (tenant TEXT NOT NULL, conversation_id TEXT NOT NULL, seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')), content TEXT NOT NULL, created_at REAL NOT NULL,
    PRIMARY KEY (tenant, conversation_id, seq));
CREATE TABLE memories (memory_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, conversation_id TEXT NOT NULL,
    memory_type TEXT NOT NULL, content TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('user_stated', 'model_inferred')),
    confidence REAL NOT NULL, source_from INTEGER NOT NULL, source_to INTEGER NOT NULL,
    job_id TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE compaction_jobs (job_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, conversation_id TEXT NOT NULL,
    through_seq INTEGER NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, claimed_at REAL,
    error TEXT, created_at REAL NOT NULL, UNIQUE (tenant, conversation_id, through_seq));
INSERT INTO conversations VALUES ('acme', 'c1', 2, 0, 1000.0);
INSERT INTO turns VALUES ('acme', 'c1', 1, 'user', 'legacy LEGACY-MARK one', 1000.0);
INSERT INTO turns VALUES ('acme', 'c1', 2, 'assistant', 'legacy LEGACY-MARK two', 1000.0);
INSERT INTO memories VALUES ('m1', 'acme', 'c1', 'decision', 'legacy LEGACY-MARK memory', 'user_stated', 1.0, 1, 1,
    'j1', 1000.0);
"""


def legacy_db(path):
    con = sqlite3.connect(path)
    con.executescript(LEGACY_SCHEMA)
    con.commit()
    con.close()


def test_a_legacy_database_opens_and_its_plaintext_rows_are_refused_until_migrated(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_db(path)
    s = make(path)  # the schema gains the encrypted flag; existing rows are flagged as plaintext
    with pytest.raises(MemoryIntegrityError, match="memory-migrate"):
        s.turns_between("acme", "c1", 0, 2)


def test_migration_encrypts_legacy_rows_in_place_and_leaves_no_plaintext_on_disk(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_db(path)
    assert b"LEGACY-MARK" in file_bytes(path)
    s = make(path)
    counts = s.migrate_plaintext(batch_size=1)  # several batches exercise the resumable loop
    assert counts == {"turns": 2, "memories": 1}
    assert [t.content for t in s.turns_between("acme", "c1", 0, 2)] == ["legacy LEGACY-MARK one",
                                                                       "legacy LEGACY-MARK two"]
    assert s.memories("acme", "c1")[0].content == "legacy LEGACY-MARK memory"
    s.close()
    assert b"LEGACY-MARK" not in file_bytes(path)


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_db(path)
    s = make(path)
    assert s.migrate_plaintext() == {"turns": 2, "memories": 1}
    assert s.migrate_plaintext() == {"turns": 0, "memories": 0}
    assert s.turns_between("acme", "c1", 0, 2)[1].content == "legacy LEGACY-MARK two"


def test_migration_needs_the_key_and_a_plaintext_store_cannot_read_encrypted_rows(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_db(path)
    with pytest.raises(RuntimeError, match="TenantCipher"):
        ConversationStore(path, plaintext=True).migrate_plaintext()
    s = make(path)
    s.migrate_plaintext()
    s.close()
    plain = ConversationStore(path, plaintext=True)
    with pytest.raises(MemoryIntegrityError, match="no cipher"):
        plain.turns_between("acme", "c1", 0, 2)
