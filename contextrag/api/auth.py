"""API-key authentication that *binds* the caller to a tenant, a set of ACL groups and a role.

Without this the tenant (``X-Tenant``) and groups (request body) are asserted by the caller, i.e. anyone
can read anyone's data. With a key store, they are derived from the key and any conflicting claim is refused.

Keys file (``CONTEXTRAG_KEYS_FILE``), only SHA-256 digests are stored::

    {"<sha256 hex of key>": {"id": "ana", "tenant": "acme", "groups": ["legal"], "role": "reader"}}

roles: ``reader`` (query), ``admin`` (query + ingest + delete + audit for its own tenant), ``ops`` (/metrics only).
Generate entries with ``python -m contextrag keygen``.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

log = logging.getLogger("contextrag.auth")
ROLES = ("reader", "admin", "ops")


@dataclass(frozen=True)
class Principal:
    id: str
    tenant: str
    groups: frozenset[str]
    role: str
    expires_at: float | None = None  # epoch seconds; None = no expiry

    def can(self, action: str) -> bool:
        allowed = {"reader": {"query"}, "admin": {"query", "ingest", "delete", "audit", "list", "erase"}, "ops": {"metrics"}}
        return action in allowed[self.role]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> str:
    return "crag_" + secrets.token_urlsafe(32)


def new_entry(key_id: str, tenant: str, role: str = "reader", groups: tuple[str, ...] = (),
              expires_at: str | None = None) -> tuple[str, str, dict]:
    """Returns (plaintext key to hand to the user, digest, JSON entry to store). ``expires_at`` is ISO-8601 with a timezone."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    key = generate_key()
    entry = {"id": key_id, "tenant": tenant, "groups": list(groups), "role": role}
    if expires_at:
        entry["expires_at"] = expires_at
    return key, hash_key(key), entry


class KeyStoreUnavailable(RuntimeError):
    """The keys file cannot be read or parsed. Authentication fails closed while this holds."""


class KeyStore:
    """API keys bound to a tenant, groups and a role. Stored as SHA-256 digests.

    When built from a file, the file is re-read whenever its modification time changes, so a key removed from the file stops
    working on the next request without a restart. An unreadable or invalid file fails closed: no request authenticates
    until the file is fixed (the alternative would keep honouring keys that were just revoked). An entry with
    ``expires_at`` in the past is refused.
    """

    def __init__(self, entries: dict[str, dict], *, path: str | None = None, clock: Callable[[], float] = time.time):
        self._path, self._clock = path, clock
        self._mtime: float | None = None
        self._entries: dict[str, Principal] = {}
        self._broken: str | None = None  # the reason the file cannot be used, reported by class only
        self._load(entries)
        if path is not None:
            self._mtime = os.stat(path).st_mtime

    def _load(self, entries: dict[str, dict]) -> None:
        loaded: dict[str, Principal] = {}
        for digest, e in entries.items():
            who = e.get("id", digest[:8])
            if e.get("role", "reader") not in ROLES:
                raise ValueError(f"invalid role for key {who}")
            if e.get("role") != "ops" and not e.get("tenant"):
                raise ValueError(f"key {who} needs a tenant")
            expires = e.get("expires_at")
            expiry = None
            if expires:
                try:
                    parsed = datetime.fromisoformat(str(expires))
                except ValueError:
                    raise ValueError(f"key {who} has an invalid expires_at") from None
                if parsed.tzinfo is None:
                    raise ValueError(f"key {who}: expires_at needs a timezone")
                expiry = parsed.timestamp()
            loaded[digest.lower()] = Principal(who, e.get("tenant", ""), frozenset(e.get("groups", [])),
                                               e.get("role", "reader"), expiry)
        self._entries = loaded

    def _refresh(self) -> None:
        if self._path is None:
            return
        try:
            mtime = os.stat(self._path).st_mtime
            if mtime == self._mtime and self._broken is None:
                return
            with open(self._path, encoding="utf-8") as f:
                entries = json.load(f)
            self._load(entries)
            self._mtime, self._broken = mtime, None
        except Exception as exc:  # the class only: messages can name paths and values
            self._broken = type(exc).__name__
            self._entries = {}
            log.error("keys file unusable (%s); authentication is refused until it is fixed", self._broken)

    def __len__(self):
        self._refresh()
        return len(self._entries)

    def authenticate(self, key: str | None) -> Principal | None:
        if not key:
            return None
        self._refresh()
        if self._broken is not None:
            return None
        digest = hash_key(key)
        found = None
        for d, p in self._entries.items():  # compare against all digests: no early exit on a match
            if hmac.compare_digest(d, digest):
                found = p
        if found is None or (found.expires_at is not None and self._clock() >= found.expires_at):
            return None
        return found

    @classmethod
    def from_file(cls, path: str) -> "KeyStore":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f), path=path)

    @classmethod
    def from_env(cls) -> "KeyStore | None":
        path = os.getenv("CONTEXTRAG_KEYS_FILE")
        return cls.from_file(path) if path else None
