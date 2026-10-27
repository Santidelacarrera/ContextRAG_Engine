"""API key lifecycle: revocation and expiry take effect without a restart; an unusable keys file fails closed."""
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from contextrag import factory
from contextrag.api.auth import KeyStore, new_entry
from contextrag.api.server import create_app


def write(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")
    # make the change visible to mtime even on filesystems with coarse timestamps
    stamp = os.stat(path).st_mtime + 1
    os.utime(path, (stamp, stamp))


@pytest.fixture
def keyfile(tmp_path):
    key, digest, entry = new_entry("ana", "acme", "admin", ("caso-a",))
    path = tmp_path / "keys.json"
    write(path, {digest: entry})
    client = TestClient(create_app(factory.offline(), keystore=KeyStore.from_file(str(path))))
    return path, client, key, digest, entry


def test_a_key_removed_from_the_file_stops_working_without_a_restart(keyfile):
    path, client, key, _, _ = keyfile
    assert client.get("/documents", headers={"X-API-Key": key}).status_code == 200
    write(path, {})  # revoked by the operator
    assert client.get("/documents", headers={"X-API-Key": key}).status_code == 401


def test_a_role_downgrade_takes_effect_on_the_next_request(keyfile):
    path, client, key, digest, entry = keyfile
    assert client.get("/documents", headers={"X-API-Key": key}).status_code == 200
    write(path, {digest: {**entry, "role": "reader"}})
    assert client.get("/documents", headers={"X-API-Key": key}).status_code == 403  # reader may not list


def test_an_expired_key_is_refused_and_a_valid_one_is_not(tmp_path):
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    k_old, d_old, e_old = new_entry("old", "acme", "reader", (), expires_at=past)
    k_new, d_new, e_new = new_entry("new", "acme", "reader", (), expires_at=future)
    ks = KeyStore({d_old: e_old, d_new: e_new})
    assert ks.authenticate(k_old) is None
    assert ks.authenticate(k_new).id == "new"


def test_an_expiry_without_a_timezone_or_in_an_invalid_format_is_refused_at_load():
    _, digest, entry = new_entry("x", "acme", "reader", ())
    for bad in ("2030-01-01T00:00:00", "next tuesday"):
        with pytest.raises(ValueError, match="expires_at"):
            KeyStore({digest: {**entry, "expires_at": bad}})


def test_an_unreadable_keys_file_fails_closed_and_recovers_when_fixed(keyfile, caplog):
    path, client, key, digest, entry = keyfile
    path.write_text("{ not json", encoding="utf-8")
    stamp = os.stat(path).st_mtime + 2
    os.utime(path, (stamp, stamp))
    with caplog.at_level(logging.ERROR, logger="contextrag.auth"):
        assert client.get("/documents", headers={"X-API-Key": key}).status_code == 401
    assert "JSONDecodeError" in caplog.text and "not json" not in caplog.text  # the class, never the content
    write(path, {digest: entry})
    assert client.get("/documents", headers={"X-API-Key": key}).status_code == 200


def test_a_missing_keys_file_fails_closed(keyfile):
    path, client, key, _, _ = keyfile
    os.remove(path)
    assert client.get("/documents", headers={"X-API-Key": key}).status_code == 401


def test_an_unchanged_file_is_not_reread_on_every_request(keyfile, monkeypatch):
    path, client, key, _, _ = keyfile
    reads = {"n": 0}
    import contextrag.api.auth as A

    real_open = open

    def counting_open(file, *a, **k):
        if str(file) == str(path):
            reads["n"] += 1
        return real_open(file, *a, **k)

    monkeypatch.setattr(A, "open", counting_open, raising=False)
    for _ in range(20):
        client.get("/documents", headers={"X-API-Key": key})
    assert reads["n"] == 0  # the stat check is enough while nothing changed


def test_a_key_loaded_from_memory_is_unaffected_by_the_lifecycle_rules():
    k, d, e = new_entry("mem", "acme", "reader", ())
    assert KeyStore({d: e}).authenticate(k).id == "mem"
