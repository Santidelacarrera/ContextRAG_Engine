"""Tenant isolation helpers and optional at-rest encryption of chunk text.

Vectors must stay in the clear to be searchable; what is encrypted is the text
payload, with a key derived per tenant from a master key.
"""
from __future__ import annotations

import base64
import hashlib
import hmac


class TenantCipher:
    def __init__(self, master_key: bytes):
        if len(master_key) < 16:
            raise ValueError("master key must be at least 16 bytes")
        self._master = master_key

    def _fernet(self, tenant: str):
        from cryptography.fernet import Fernet  # optional dependency

        key = hmac.new(self._master, f"tenant:{tenant}".encode(), hashlib.sha256).digest()
        return Fernet(base64.urlsafe_b64encode(key))

    def encrypt(self, tenant: str, text: str) -> str:
        return self._fernet(tenant).encrypt(text.encode()).decode()

    def decrypt(self, tenant: str, token: str) -> str:
        return self._fernet(tenant).decrypt(token.encode()).decode()


def visible(acl: tuple[str, ...] | list[str], groups: frozenset[str] | set[str]) -> bool:
    """Empty ACL => tenant-wide; otherwise the caller needs at least one matching group."""
    return not acl or bool(set(acl) & set(groups))
