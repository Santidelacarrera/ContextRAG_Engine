"""MASTER_KEY rotation: data stays readable at every step, an interrupted run loses nothing, and the old key can be retired."""
import json
import logging
import sqlite3

import pytest
from cryptography.fernet import InvalidToken

from contextrag.__main__ import main
from contextrag.core.cache import EncryptedCache, InMemoryCache
from contextrag.core.chunking import chunk_document
from contextrag.core.config import Settings
from contextrag.core.embeddings import HashingEmbedder
from contextrag.core.security import TenantCipher
from contextrag.memory.store import ConversationStore, MemoryDraft, MemoryIntegrityError
from contextrag.retrieval.vector_store import QdrantVectorStore

OLD = b"old-master-key-0123456789abcdef-OLD"
NEW = b"new-master-key-0123456789abcdef-NEW"
SECRET_TEXT = "El cliente Pedro Soto reconoce una deuda de 80 millones con la sociedad."


# ---- the cipher --------------------------------------------------------------------------------------------

def test_a_token_made_under_the_old_key_opens_with_new_plus_previous_but_not_with_new_alone():
    old_token = TenantCipher(OLD).encrypt("acme", SECRET_TEXT)
    assert TenantCipher(NEW, [OLD]).decrypt("acme", old_token) == SECRET_TEXT
    with pytest.raises(InvalidToken):
        TenantCipher(NEW).decrypt("acme", old_token)


def test_new_encryptions_use_the_primary_key_only():
    token = TenantCipher(NEW, [OLD]).encrypt("acme", SECRET_TEXT)
    assert TenantCipher(NEW).decrypt("acme", token) == SECRET_TEXT  # readable once the old key is retired
    with pytest.raises(InvalidToken):
        TenantCipher(OLD).decrypt("acme", token)


def test_needs_rotation_distinguishes_old_new_and_unreadable():
    c = TenantCipher(NEW, [OLD])
    assert c.needs_rotation("acme", TenantCipher(OLD).encrypt("acme", "x")) is True
    assert c.needs_rotation("acme", c.encrypt("acme", "x")) is False
    with pytest.raises(InvalidToken):  # a token no configured key opens is an error, not "pending rotation"
        c.needs_rotation("acme", TenantCipher(b"some-other-key-0123456789abcdef").encrypt("acme", "x"))


def test_tenant_separation_survives_rotation():
    c = TenantCipher(NEW, [OLD])
    token = TenantCipher(OLD).encrypt("acme", SECRET_TEXT)
    with pytest.raises(InvalidToken):
        c.decrypt("rival", token)


def test_key_material_never_appears_in_repr_or_errors(caplog):
    with caplog.at_level(logging.DEBUG):
        c = TenantCipher(b"short-but-ok-0123456789")
    assert "short-but-ok" not in repr(c) and "short-but-ok" not in caplog.text
    with pytest.raises(ValueError) as info:
        TenantCipher(b"tooshort")
    assert "tooshort" not in str(info.value)
    with pytest.raises(ValueError) as info:
        TenantCipher(NEW, [b"tiny"])
    assert "tiny" not in str(info.value) and "#1" in str(info.value)


def test_a_key_shorter_than_the_recommended_length_is_warned_about(caplog):
    with caplog.at_level(logging.WARNING, logger="contextrag.security"):
        TenantCipher(b"0123456789abcdef")  # valid (16 bytes) but short
    assert "shorter than 32" in caplog.text


# ---- conversation memory --------------------------------------------------------------------------------------

def seeded_store(path, cipher):
    s = ConversationStore(path, cipher=cipher)
    s.create_conversation("acme", "c1")
    for i in range(4):
        s.append_turn("acme", "c1", "user" if i % 2 == 0 else "assistant", f"{SECRET_TEXT} (turno {i})")
    job = s.plan_compaction("acme", "c1", threshold_tokens=0, keep_recent_turns=1)
    claim = s.claim_job(job, lease_seconds=60)
    s.complete_job(claim, [MemoryDraft("decision", "Decidimos pagar en cuotas.", "user_stated", 1.0, 1)])
    return s


def test_memory_rotation_end_to_end(tmp_path):
    path = tmp_path / "memory.db"
    seeded_store(path, TenantCipher(OLD)).close()

    s = ConversationStore(path, cipher=TenantCipher(NEW, [OLD]))
    assert len(s.turns_between("acme", "c1", 0, 4)) == 4  # step 1: new primary, old kept: everything is readable
    assert s.rekey(dry_run=True) == {"turns": 4, "memories": 1}
    assert s.rekey(dry_run=True) == {"turns": 4, "memories": 1}  # a dry run changes nothing
    assert s.rekey() == {"turns": 4, "memories": 1}  # step 2: re-encrypt
    assert s.rekey() == {"turns": 0, "memories": 0}  # idempotent
    s.close()

    only_new = ConversationStore(path, cipher=TenantCipher(NEW))  # step 3: the old key is retired
    assert [t.content for t in only_new.turns_between("acme", "c1", 0, 4)][0].startswith(SECRET_TEXT)
    assert only_new.memories("acme", "c1")[0].content == "Decidimos pagar en cuotas."
    only_new.close()
    raw = sqlite3.connect(path).execute("SELECT group_concat(content) FROM turns").fetchone()[0]
    assert "Pedro Soto" not in raw


def test_an_interrupted_run_loses_nothing_and_resumes(tmp_path, monkeypatch):
    path = tmp_path / "memory.db"
    seeded_store(path, TenantCipher(OLD)).close()
    s = ConversationStore(path, cipher=TenantCipher(NEW, [OLD]))

    real = s._store_text
    calls = {"n": 0}

    def flaky(tenant, label, text):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("process killed")  # dies in the middle of the third row
        return real(tenant, label, text)

    monkeypatch.setattr(s, "_store_text", flaky)
    with pytest.raises(RuntimeError):
        s.rekey(batch_size=1)  # earlier rows were committed in their own transactions; this one rolled back
    monkeypatch.undo()

    assert len(s.turns_between("acme", "c1", 0, 4)) == 4  # every row is still readable, old or new
    remaining = s.rekey(dry_run=True)
    assert 0 < remaining["turns"] + remaining["memories"] < 5  # some rows done, some pending
    s.rekey()
    assert s.rekey(dry_run=True) == {"turns": 0, "memories": 0}
    s.close()
    ConversationStore(path, cipher=TenantCipher(NEW)).turns_between("acme", "c1", 0, 4)  # readable with the new key alone


def test_rekey_stops_before_touching_a_row_that_no_key_opens(tmp_path):
    path = tmp_path / "memory.db"
    seeded_store(path, TenantCipher(OLD)).close()
    wrong = ConversationStore(path, cipher=TenantCipher(b"completely-different-key-0123456789"))  # old key forgotten
    before = sqlite3.connect(path).execute("SELECT group_concat(content) FROM turns").fetchone()[0]
    with pytest.raises(MemoryIntegrityError):
        wrong.rekey()
    wrong.close()
    assert sqlite3.connect(path).execute("SELECT group_concat(content) FROM turns").fetchone()[0] == before


def test_rekey_needs_a_cipher():
    with pytest.raises(RuntimeError):
        ConversationStore(":memory:", plaintext=True).rekey()


# ---- chunk text in Qdrant ---------------------------------------------------------------------------------------

def qdrant_pair():
    from qdrant_client import QdrantClient

    client = QdrantClient(":memory:")
    emb = HashingEmbedder()
    old_store = QdrantVectorStore(emb.dim, client=client, cipher=TenantCipher(OLD))
    parents, children = chunk_document(SECRET_TEXT + " Otra frase del documento.", doc_id="d.md", tenant="acme",
                                       source="d.md", kind="text", acl=(), page_starts=None, settings=Settings())
    old_store.add(parents, children, emb.embed([c.text for c in children]))
    return client, emb, parents


def test_qdrant_rotation_end_to_end():
    client, emb, parents = qdrant_pair()
    rotating = QdrantVectorStore(emb.dim, client=client, cipher=TenantCipher(NEW, [OLD]))
    assert rotating.parent(parents[0].id, "acme").text == parents[0].text  # readable under new + previous
    pending = rotating.rekey(dry_run=True)
    assert pending["children"] > 0 and pending["parents"] > 0
    assert rotating.rekey(dry_run=True) == pending
    assert rotating.rekey() == pending
    assert rotating.rekey() == {"children": 0, "parents": 0}
    only_new = QdrantVectorStore(emb.dim, client=client, cipher=TenantCipher(NEW))
    assert only_new.parent(parents[0].id, "acme").text == parents[0].text  # the old key is no longer needed
    assert [c.text for c in only_new.iter_children("acme")]


def test_qdrant_rekey_refuses_text_no_key_opens_and_changes_nothing():
    client, emb, parents = qdrant_pair()
    stranger = QdrantVectorStore(emb.dim, client=client, cipher=TenantCipher(b"another-key-0123456789abcdef-XYZ"))
    with pytest.raises(InvalidToken):
        stranger.rekey()
    assert QdrantVectorStore(emb.dim, client=client, cipher=TenantCipher(OLD)).parent(parents[0].id, "acme").text == parents[0].text


# ---- the command ------------------------------------------------------------------------------------------------

def test_cli_dry_run_then_rotation_then_retire_the_old_key(tmp_path, monkeypatch, capsys):
    path = tmp_path / "memory.db"
    seeded_store(path, TenantCipher(OLD)).close()
    monkeypatch.setenv("MEMORY_DB", str(path))
    monkeypatch.setenv("MASTER_KEY", NEW.decode())
    monkeypatch.setenv("MASTER_KEY_PREVIOUS", OLD.decode())
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert main(["rotate-key", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out) == {"dry_run": True, "memory": {"turns": 4, "memories": 1}}
    assert main(["rotate-key"]) == 0
    assert json.loads(capsys.readouterr().out)["memory"] == {"turns": 4, "memories": 1}
    monkeypatch.delenv("MASTER_KEY_PREVIOUS")
    ConversationStore(path, cipher=TenantCipher(NEW)).turns_between("acme", "c1", 0, 4)


def test_cli_refuses_to_run_without_both_keys_and_prints_no_key(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MEMORY_DB", str(tmp_path / "m.db"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("MASTER_KEY", NEW.decode())
    monkeypatch.delenv("MASTER_KEY_PREVIOUS", raising=False)
    assert main(["rotate-key"]) == 2
    err = capsys.readouterr().err
    assert "MASTER_KEY_PREVIOUS" in err and NEW.decode() not in err


def test_cli_failure_message_names_the_class_not_the_data(tmp_path, monkeypatch, capsys):
    path = tmp_path / "memory.db"
    seeded_store(path, TenantCipher(OLD)).close()
    monkeypatch.setenv("MEMORY_DB", str(path))
    monkeypatch.setenv("MASTER_KEY", NEW.decode())
    monkeypatch.setenv("MASTER_KEY_PREVIOUS", "wrong-previous-key-0123456789abcdef")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert main(["rotate-key"]) == 1
    captured = capsys.readouterr()
    assert "MemoryIntegrityError" in captured.err and "Pedro Soto" not in captured.err + captured.out


def test_factory_refuses_previous_keys_without_a_primary(monkeypatch):
    from contextrag import factory

    monkeypatch.delenv("MASTER_KEY", raising=False)
    monkeypatch.setenv("MASTER_KEY_PREVIOUS", OLD.decode())
    with pytest.raises(RuntimeError, match="MASTER_KEY_PREVIOUS is set but MASTER_KEY is not"):
        factory.from_env()


# ---- encrypted cache ------------------------------------------------------------------------------------------------

def test_the_cache_backend_never_sees_plaintext_and_counters_keep_working():
    backing = InMemoryCache()
    cache = EncryptedCache(backing, TenantCipher(NEW))
    cache.set("ans:abc", json.dumps({"answer": SECRET_TEXT}))
    assert SECRET_TEXT not in backing.get("ans:abc") and "Pedro" not in backing.get("ans:abc")
    assert json.loads(cache.get("ans:abc"))["answer"] == SECRET_TEXT
    assert cache.incr("epoch:acme") == 1 and cache.get("epoch:acme") == "1"  # counters stay plain and readable
    assert cache.health() is True


def test_unreadable_cache_entries_are_misses_not_errors():
    backing = InMemoryCache()
    backing.set("ans:legacy", json.dumps({"answer": "cached before encryption"}))  # plaintext from an older deployment
    backing.set("ans:other", TenantCipher(OLD).encrypt("__cache__", "from a retired key"))
    cache = EncryptedCache(backing, TenantCipher(NEW))
    assert cache.get("ans:legacy") is None and cache.get("ans:other") is None and cache.get("ans:missing") is None
    assert cache.unreadable == 2


def test_cached_answers_stay_valid_across_a_rotation_that_keeps_the_previous_key():
    backing = InMemoryCache()
    EncryptedCache(backing, TenantCipher(OLD)).set("ans:k", "v")
    assert EncryptedCache(backing, TenantCipher(NEW, [OLD])).get("ans:k") == "v"


def test_qdrant_rekey_aborts_on_a_point_without_payload_and_a_repaired_rerun_completes():
    from qdrant_client.models import PointStruct

    client, emb, parents = qdrant_pair()
    rotating = QdrantVectorStore(emb.dim, client=client, cipher=TenantCipher(NEW, [OLD]))
    bad_id = "00000000-0000-0000-0000-00000000bad1"
    client.upsert(rotating.col, [PointStruct(id=bad_id, vector=[0.0] * emb.dim, payload=None)])
    with pytest.raises(RuntimeError, match="no tenant/text payload") as info:
        rotating.rekey()
    assert bad_id in str(info.value)  # the id is named ...
    assert SECRET_TEXT not in str(info.value)  # ... and no chunk text is
    client.delete(rotating.col, [bad_id])  # the operator removes or repairs the point
    rotating.rekey()  # finishes what the aborted run had left pending
    assert rotating.rekey() == {"children": 0, "parents": 0}  # a second run is a no-op
    only_new = QdrantVectorStore(emb.dim, client=client, cipher=TenantCipher(NEW))
    assert only_new.parent(parents[0].id, "acme").text == parents[0].text
