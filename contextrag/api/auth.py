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
import os
import secrets
from dataclasses import dataclass

ROLES = ("reader", "admin", "ops")


@dataclass(frozen=True)
class Principal:
    id: str
    tenant: str
    groups: frozenset[str]
    role: str

    def can(self, action: str) -> bool:
        allowed = {"reader": {"query"}, "admin": {"query", "ingest", "delete", "audit"}, "ops": {"metrics"}}
        return action in allowed[self.role]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def generate_key() -> str:
    return "crag_" + secrets.token_urlsafe(32)


def new_entry(key_id: str, tenant: str, role: str = "reader", groups: tuple[str, ...] = ()) -> tuple[str, str, dict]:
    """Returns (plaintext key to hand to the user, digest, JSON entry to store)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    key = generate_key()
    return key, hash_key(key), {"id": key_id, "tenant": tenant, "groups": list(groups), "role": role}


class KeyStore:
    def __init__(self, entries: dict[str, dict]):
        self._entries: dict[str, Principal] = {}
        for digest, e in entries.items():
            if e.get("role", "reader") not in ROLES:
                raise ValueError(f"invalid role for key {e.get('id', digest[:8])}")
            if e.get("role") != "ops" and not e.get("tenant"):
                raise ValueError(f"key {e.get('id', digest[:8])} needs a tenant")
            self._entries[digest.lower()] = Principal(
                e.get("id", digest[:8]), e.get("tenant", ""), frozenset(e.get("groups", [])), e.get("role", "reader"))

    def __len__(self):
        return len(self._entries)

    def authenticate(self, key: str | None) -> Principal | None:
        if not key:
            return None
        digest = hash_key(key)
        found = None
        for d, p in self._entries.items():  # compare against all digests: no early exit on a match
            if hmac.compare_digest(d, digest):
                found = p
        return found

    @classmethod
    def from_file(cls, path: str) -> "KeyStore":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def from_env(cls) -> "KeyStore | None":
        path = os.getenv("CONTEXTRAG_KEYS_FILE")
        return cls.from_file(path) if path else None
