"""Tenant isolation helpers and optional at-rest encryption of text.

Vectors must stay in the clear to be searchable; what is encrypted is the text payload, with a key derived per tenant
from a master key (HMAC-SHA256 of ``tenant:<name>``, used as a Fernet key: AES-128-CBC with HMAC-SHA256, authenticated).

Key management
--------------
* ``MASTER_KEY``: at least 16 bytes, 32 or more recommended. Generate one with
  ``python -c "import secrets; print(secrets.token_urlsafe(32))"`` and keep it in a secret manager, never in the
  repository, an image or a log. There is no copy of it in the data: losing it makes encrypted data unreadable, so
  back it up separately from the data it protects.
* Rotation without a window of unreadable data: start with the new key as ``MASTER_KEY`` and the old one(s) in
  ``MASTER_KEY_PREVIOUS`` (comma-separated). Everything stays readable (decryption tries every key, encryption uses the
  primary), then ``python -m contextrag rotate-key`` re-encrypts what is still under an old key. Only after that, and after
  backups made under the old key are no longer needed, is the old key dropped. A backup is readable only with the key
  that was primary when it was taken.
* What this does not protect: metadata (document names, ids, section headings, timestamps), vectors, and anything
  outside the stores that use this cipher. See the README for the full list.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from collections.abc import Sequence

log = logging.getLogger("contextrag.security")
MIN_KEY_BYTES = 16
RECOMMENDED_KEY_BYTES = 32


class TenantCipher:
    def __init__(self, master_key: bytes, previous: Sequence[bytes] = ()):
        keys = [master_key, *previous]
        for i, key in enumerate(keys):
            if len(key) < MIN_KEY_BYTES:
                # the message names the position of the key, never its value
                raise ValueError(f"master key #{i} must be at least {MIN_KEY_BYTES} bytes")
        if len(master_key) < RECOMMENDED_KEY_BYTES:
            log.warning("MASTER_KEY is shorter than %d bytes; use a random key of at least that length",
                        RECOMMENDED_KEY_BYTES)
        self._masters = keys

    def __repr__(self) -> str:  # never show key material, even by accident in a log or a traceback
        return f"TenantCipher(keys={len(self._masters)})"

    @staticmethod
    def _fernet(master: bytes, tenant: str):
        from cryptography.fernet import Fernet  # optional dependency

        key = hmac.new(master, f"tenant:{tenant}".encode(), hashlib.sha256).digest()
        return Fernet(base64.urlsafe_b64encode(key))

    def _multi(self, tenant: str):
        from cryptography.fernet import MultiFernet

        return MultiFernet([self._fernet(m, tenant) for m in self._masters])

    def encrypt(self, tenant: str, text: str) -> str:
        """Always under the primary key."""
        return self._multi(tenant).encrypt(text.encode()).decode()

    def decrypt(self, tenant: str, token: str) -> str:
        """Under the primary or any previous key. Raises ``InvalidToken`` for any other key or a modified token."""
        return self._multi(tenant).decrypt(token.encode()).decode()

    def needs_rotation(self, tenant: str, token: str) -> bool:
        """True when ``token`` decrypts only under a previous key. Raises if it decrypts under none."""
        from cryptography.fernet import InvalidToken

        try:
            self._fernet(self._masters[0], tenant).decrypt(token.encode())
            return False
        except InvalidToken:
            self.decrypt(tenant, token)  # raises InvalidToken if no key opens it: unreadable is not "needs rotation"
            return True


def visible(acl: tuple[str, ...] | list[str], groups: frozenset[str] | set[str]) -> bool:
    """Empty ACL => tenant-wide; otherwise the caller needs at least one matching group."""
    return not acl or bool(set(acl) & set(groups))


def load_master_keys(env: "dict[str, str] | None" = None) -> tuple[bytes | None, list[bytes]]:
    """(primary key, previous keys) from the environment, preferring files over variables.

    ``MASTER_KEY_FILE`` / ``MASTER_KEY_PREVIOUS_FILE`` point at files mounted by the orchestrator (a Docker or Kubernetes
    secret). They are not visible in ``docker inspect`` or the process environment, and they are refused unless only the
    owner can read them. Setting a variable and its file form together is an error, so the operator cannot believe a key
    is in use when another one is.
    """
    import os
    import stat

    e = os.environ if env is None else env

    def read(name: str) -> str | None:
        var, path = e.get(name), e.get(name + "_FILE")
        if var and path:
            raise ValueError(f"set {name} or {name}_FILE, not both")
        if path:
            st = os.stat(path)
            if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                raise ValueError(f"{name}_FILE must not be readable by group or others (use mode 0600)")
            with open(path, encoding="utf-8") as f:
                return f.read()
        return var

    primary = (read("MASTER_KEY") or "").strip()
    previous_raw = read("MASTER_KEY_PREVIOUS") or ""
    previous = [k.strip().encode() for k in previous_raw.replace("\n", ",").split(",") if k.strip()]
    if previous and not primary:
        raise ValueError("MASTER_KEY_PREVIOUS is set but MASTER_KEY is not: the primary key is required")
    return (primary.encode() if primary else None), previous
