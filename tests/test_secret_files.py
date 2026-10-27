"""Master keys from files: loaded, permission-checked, never mixed with their variable form, and never echoed."""
import os

import pytest

from contextrag.core.security import TenantCipher, load_master_keys

KEY = "0123456789abcdef0123456789abcdef-PRIMARY"
OLD = "fedcba9876543210fedcba9876543210-RETIRED"


def secret(tmp_path, name, text, mode=0o600):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    os.chmod(p, mode)
    return str(p)


def test_keys_are_read_from_files_with_a_trailing_newline_removed(tmp_path):
    env = {"MASTER_KEY_FILE": secret(tmp_path, "k", KEY + "\n"),
           "MASTER_KEY_PREVIOUS_FILE": secret(tmp_path, "p", OLD + "\n")}
    primary, previous = load_master_keys(env)
    assert primary == KEY.encode() and previous == [OLD.encode()]
    TenantCipher(primary, previous)  # usable as-is


def test_a_key_file_readable_by_others_is_refused_and_the_key_is_not_in_the_message(tmp_path):
    env = {"MASTER_KEY_FILE": secret(tmp_path, "k", KEY, mode=0o644)}
    with pytest.raises(ValueError, match="must not be readable") as info:
        load_master_keys(env)
    assert KEY not in str(info.value)


def test_a_group_readable_file_is_refused_too(tmp_path):
    with pytest.raises(ValueError, match="must not be readable"):
        load_master_keys({"MASTER_KEY_FILE": secret(tmp_path, "k", KEY, mode=0o640)})


def test_the_variable_and_the_file_together_are_refused_so_the_active_key_is_never_ambiguous(tmp_path):
    env = {"MASTER_KEY": KEY, "MASTER_KEY_FILE": secret(tmp_path, "k", KEY)}
    with pytest.raises(ValueError, match="not both"):
        load_master_keys(env)


def test_a_previous_key_without_a_primary_is_refused(tmp_path):
    with pytest.raises(ValueError, match="primary key is required"):
        load_master_keys({"MASTER_KEY_PREVIOUS": OLD})


def test_the_variable_form_still_works_for_development(tmp_path):
    assert load_master_keys({"MASTER_KEY": KEY}) == (KEY.encode(), [])


def test_the_factory_builds_the_cipher_from_a_file(tmp_path, monkeypatch):
    from contextrag import factory

    monkeypatch.delenv("MASTER_KEY", raising=False)
    monkeypatch.setenv("MASTER_KEY_FILE", secret(tmp_path, "k", KEY))
    monkeypatch.delenv("MEMORY_DB", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    rag = factory.from_env()
    assert isinstance(rag.index.store.cipher, TenantCipher)  # the file's key is the one in use
    rag.close()


def test_rotate_key_uses_the_same_loader(tmp_path, monkeypatch, capsys):
    from contextrag.__main__ import main

    monkeypatch.delenv("MASTER_KEY", raising=False)
    monkeypatch.setenv("MASTER_KEY_FILE", secret(tmp_path, "k", KEY, mode=0o644))
    assert main(["rotate-key", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "must not be readable" in err and KEY not in err
