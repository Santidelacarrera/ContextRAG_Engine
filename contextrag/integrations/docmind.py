"""Reference adapter for DOCMIND_AI: the calls DOCMIND needs, with identity checked before every operation.

This file is a reference for DOCMIND_AI's own code. It is not DOCMIND code and nothing here has run against DOCMIND.

Identity. DOCMIND authenticates its user with its own mechanism and passes the result as a :class:`VerifiedPrincipal`.
That object must come from DOCMIND's session or key store, never from a request body or a header a browser sent. The
adapter then asks DOCMIND's :class:`CredentialDirectory` for the ContextRAG credential that user owns. Before acting, it
calls ``GET /me`` with that credential and refuses to continue unless the key belongs to the expected organisation
(tenant) and is the key DOCMIND recorded for that user. A mismatch fails closed; the adapter never falls back to another
credential.

Authorisation is enforced by ContextRAG on the server (the key's groups and role). The adapter does not duplicate it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from ..client import ContextRAGClient, ContextRAGError


class AccessDenied(Exception):
    """DOCMIND has no active ContextRAG credential for this user. Nothing was sent to ContextRAG."""


class IdentityMismatch(Exception):
    """The credential DOCMIND selected does not belong to the expected organisation or key. Nothing was performed."""


@dataclass(frozen=True)
class VerifiedPrincipal:
    organization_id: str  # the tenant; DOCMIND's identifier for the organisation
    user_id: str          # DOCMIND's identifier for the user, used by DOCMIND for its own audit


@dataclass(frozen=True)
class Credential:
    key: str     # the ContextRAG API key; a secret, never logged
    key_id: str  # the ``id`` of the key's entry in CONTEXTRAG_KEYS_FILE, recorded by DOCMIND when it issued the key


class CredentialDirectory(Protocol):
    """DOCMIND's own lookup. Returns ``None`` when the user has no active ContextRAG credential (revoked, expired, never
    issued, or the user left the organisation)."""

    def credential_for(self, who: VerifiedPrincipal) -> Credential | None: ...


class DocmindRagAdapter:
    def __init__(self, base_url: str, directory: CredentialDirectory, *, timeout_seconds: float = 30.0):
        self._base = base_url
        self._directory = directory
        self._timeout = timeout_seconds

    def _client(self, who: VerifiedPrincipal) -> ContextRAGClient:
        if not who.organization_id or not who.user_id:
            raise AccessDenied("the principal must carry an organisation and a user")
        cred = self._directory.credential_for(who)
        if cred is None:
            raise AccessDenied("no active ContextRAG credential for this user")
        client = ContextRAGClient(self._base, cred.key, timeout_seconds=self._timeout)
        me = client.me()  # 401 here means the credential was revoked or expired: the error is passed on, not hidden
        if me.get("tenant") != who.organization_id:
            raise IdentityMismatch("the credential belongs to another organisation")
        if me.get("key_id") != cred.key_id:
            raise IdentityMismatch("the credential is not the one recorded for this user")
        return client

    # -- operations: each one re-verifies the identity first -------------------------------------------------------
    def ingest(self, who: VerifiedPrincipal, documents: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        return self._client(who).ingest(documents)

    def query(self, who: VerifiedPrincipal, question: str, *, groups: Iterable[str] = (),
              conversation_id: str | None = None) -> dict[str, Any]:
        return self._client(who).query(question, groups=groups, conversation_id=conversation_id)

    def delete_document(self, who: VerifiedPrincipal, document_id: str) -> dict[str, Any]:
        return self._client(who).delete_document(document_id)

    def list_documents(self, who: VerifiedPrincipal) -> dict[str, Any]:
        return self._client(who).list_documents()

    def audit(self, who: VerifiedPrincipal, limit: int = 50) -> list[dict[str, Any]]:
        return self._client(who).audit(limit)


__all__ = ["AccessDenied", "IdentityMismatch", "VerifiedPrincipal", "Credential", "CredentialDirectory",
           "DocmindRagAdapter", "ContextRAGError"]
