"""Reference HTTP client for the ContextRAG API (the contract in docs/INTEGRATION_DOCMIND.md).

It is a reference, not a deployed component: a consumer such as DOCMIND_AI should either vendor it or reimplement the
same calls. It uses only the standard library and follows the server's rules:

* the API key is the only identity: tenant, groups and the audit identifier come from the key on the server, never from
  this client's arguments (there is no way to name a tenant or a user here);
* no redirects are followed, a request has a timeout, and an error keeps the server's stable code, not its message;
* the key is never printed or put in an exception.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable, Mapping

MAX_DOCUMENTS_PER_REQUEST = 50  # the server refuses more; checked here so a batch is never split silently
MAX_RESPONSE_BYTES = 8_000_000


class ContextRAGError(Exception):
    """A non-2xx answer or a transport failure. ``code`` is the server's stable code when it has one."""

    def __init__(self, status: int | None, code: str):
        super().__init__(f"{status} {code}" if status is not None else code)
        self.status, self.code = status, code


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirects are not followed", headers, fp)


_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


def _server_code(raw: bytes, status: int) -> str:
    try:
        body = json.loads(raw)
    except ValueError:
        return f"http_{status}"
    for key in ("error", "detail"):  # both forms the API uses; only a stable upper-case token is passed on
        value = body.get(key) if isinstance(body, dict) else None
        if isinstance(value, str) and _CODE.match(value):
            return value
    return f"http_{status}"


class ContextRAGClient:
    def __init__(self, base_url: str, api_key: str, *, timeout_seconds: float = 30.0):
        if not (1 <= timeout_seconds <= 600):
            raise ValueError("timeout_seconds must be 1..600")
        parts = urllib.parse.urlsplit(base_url or "")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("base_url must be an http:// or https:// URL")
        if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            raise ValueError("base_url must not contain credentials, a query or a fragment")
        if not api_key:
            raise ValueError("an API key is required")
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout_seconds
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirects)

    def __repr__(self) -> str:  # never shows the key
        return f"ContextRAGClient(base_url={self._base!r})"

    # -- calls ----------------------------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._call("GET", "/healthz", auth=False)

    def ingest(self, documents: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Each document: ``doc_id`` and exactly one of ``text`` or ``content_base64``; optional ``source``, ``kind``,
        ``acl`` (list of group names) and ``force``. Re-sending a document updates it (see the contract)."""
        docs = [dict(d) for d in documents]
        if not docs or len(docs) > MAX_DOCUMENTS_PER_REQUEST:
            raise ValueError(f"send 1..{MAX_DOCUMENTS_PER_REQUEST} documents per call")
        for d in docs:
            if not d.get("doc_id") or (("text" in d) == ("content_base64" in d)):
                raise ValueError("each document needs doc_id and exactly one of text / content_base64")
        return self._call("POST", "/ingest", body={"documents": docs})

    def query(self, question: str, *, groups: Iterable[str] = (), top_k: int | None = None,
              conversation_id: str | None = None) -> dict[str, Any]:
        """``groups`` may only narrow the key's groups; asking for more is a 403 ``groups not granted``."""
        body: dict[str, Any] = {"question": question, "groups": list(groups)}
        if top_k is not None:
            body["top_k"] = top_k
        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        return self._call("POST", "/query", body=body)

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        return self._call("DELETE", "/documents/" + urllib.parse.quote(doc_id, safe="/"))

    def me(self) -> dict[str, Any]:
        """The identity behind the presented key: ``key_id``, ``tenant``, ``role``, ``groups``."""
        return self._call("GET", "/me")

    def list_documents(self) -> dict[str, Any]:
        return self._call("GET", "/documents")

    def audit(self, limit: int = 50) -> list[dict[str, Any]]:
        """The caller's tenant audit rows (a restricted key sees only its own queries). Returns a JSON list."""
        return self._call("GET", "/audit?limit=" + str(int(limit)))

    # -- transport --------------------------------------------------------------------------------
    def _call(self, method: str, path: str, *, body: Any = None, auth: bool = True) -> Any:  # object or list
        data = None if body is None else json.dumps(body).encode()
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if auth:
            headers["X-API-Key"] = self._key
        req = urllib.request.Request(self._base + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self._timeout) as r:
                raw = r.read(MAX_RESPONSE_BYTES + 1)
                status = r.status
        except urllib.error.HTTPError as e:
            raw = e.read(MAX_RESPONSE_BYTES) if hasattr(e, "read") else b""
            raise ContextRAGError(e.code, _server_code(raw, e.code)) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ContextRAGError(None, type(e).__name__) from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ContextRAGError(status, "response_too_large")
        try:
            return json.loads(raw) if raw else {}
        except ValueError:
            raise ContextRAGError(status, "malformed_response") from None
