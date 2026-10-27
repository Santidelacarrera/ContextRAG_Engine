"""Key/value cache used for embeddings and query results (in-memory or Redis)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Protocol, Sequence


log = logging.getLogger("contextrag.cache")


class Cache(Protocol):
    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl: int | None = None) -> None: ...

    def incr(self, key: str) -> int: ...

    def health(self) -> bool: ...


class InMemoryCache:
    def __init__(self):
        self._d: dict[str, tuple[str, float | None]] = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is None:
                return None
            if v[1] is not None and v[1] < time.monotonic():
                del self._d[key]
                return None
            return v[0]

    def set(self, key, value, ttl=None):
        with self._lock:
            self._d[key] = (value, time.monotonic() + ttl if ttl else None)

    def incr(self, key):
        with self._lock:
            n = int(self._d.get(key, ("0", None))[0]) + 1
            self._d[key] = (str(n), None)
            return n

    def health(self):
        return True


class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379/0", client=None, prefix: str = "contextrag:"):
        if client is None:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True)
        self._r, self._p = client, prefix

    def get(self, key):
        return self._r.get(self._p + key)

    def set(self, key, value, ttl=None):
        self._r.set(self._p + key, value, ex=ttl)

    def incr(self, key):
        return int(self._r.incr(self._p + key))

    def health(self):
        try:
            return bool(self._r.ping())
        except Exception:
            return False


class EncryptedCache:
    """Seals cache values before they reach the backing store (Redis), so cached answers, evidence passages and embedding
    vectors are not readable from the cache server, a snapshot of it, or the network path to it.

    Limits: the key is derived for the cache as a whole, not per tenant, so tenant separation in the cache is still by key
    (the tenant is part of every answer key) and not by cryptography. Cache keys themselves (hashes, and ``epoch:<tenant>``
    counters) are not encrypted. A value that cannot be opened (cached before encryption was on, or under a key that has
    been retired) is a miss: the answer is recomputed, never an error and never served unverified.
    """

    _DOMAIN = "__cache__"
    _PLAIN = ("epoch:",)  # counters are read with get() and written with incr(): they stay readable

    def __init__(self, inner: Cache, cipher):
        self._inner, self._cipher = inner, cipher
        self.unreadable = 0  # entries skipped as misses; a sustained rise means a key or a deployment problem

    def _plain(self, key: str) -> bool:
        return key.startswith(self._PLAIN)

    def get(self, key):
        value = self._inner.get(key)
        if value is None or self._plain(key):
            return value
        try:
            return self._cipher.decrypt(self._DOMAIN, value)
        except Exception:
            self.unreadable += 1
            return None

    def set(self, key, value, ttl=None):
        self._inner.set(key, value if self._plain(key) else self._cipher.encrypt(self._DOMAIN, value), ttl)

    def incr(self, key):
        return self._inner.incr(key)

    def health(self):
        return self._inner.health()


class ResilientCache:
    """Makes a cache backend optional: when it fails, the system behaves as if the cache were empty.

    Without this an unreachable Redis fails every query, embedding and ingest, although the cache is only an optimisation.

    The one thing that must not be lost is invalidation. ``incr`` bumps the epoch that makes cached answers unreachable
    after a document changes or is deleted. If that call fails, the bump is remembered, and **no cached value is read or
    written until it has been applied**. Otherwise an answer cached before the change would come back to life when the
    backend recovers, and a deleted document could still be quoted.
    """

    def __init__(self, inner: Cache):
        self._inner = inner
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self.errors = 0  # exposed as contextrag_cache_errors_total; a sustained rise means the backend is unhealthy

    def _failed(self, op: str, exc: Exception) -> None:
        self.errors += 1
        log.warning("cache %s failed: %s", op, type(exc).__name__)  # the class only; messages can carry hosts or keys

    def _settled(self) -> bool:
        """True when there are no invalidations waiting. Tries to apply them first."""
        with self._lock:
            for key in list(self._pending):
                try:
                    self._inner.incr(key)
                except Exception as exc:
                    self._failed("incr", exc)
                    return False
                self._pending.discard(key)
            return True

    def get(self, key):
        if self._pending and not self._settled():
            return None
        try:
            return self._inner.get(key)
        except Exception as exc:
            self._failed("get", exc)
            return None

    def set(self, key, value, ttl=None):
        if self._pending and not self._settled():
            return
        try:
            self._inner.set(key, value, ttl)
        except Exception as exc:
            self._failed("set", exc)

    def incr(self, key):
        try:
            return self._inner.incr(key)
        except Exception as exc:
            self._failed("incr", exc)
            with self._lock:
                self._pending.add(key)
            return 0

    def health(self):
        try:
            return bool(self._inner.health())
        except Exception:
            return False


class CachedEmbedder:
    """Wraps an Embedder; vectors are cached by sha256(model, text)."""

    def __init__(self, inner, cache: Cache, name: str | None = None, ttl: int | None = 7 * 86400):
        self.inner, self.cache, self.ttl = inner, cache, ttl
        self.dim = inner.dim
        # The vector space is the key: a cached vector from model A must never answer a request for model B.
        self.model_id = name or getattr(inner, "model_id", None) or f"{type(inner).__name__}:{inner.dim}"
        self.name = self.model_id

    def _key(self, text: str) -> str:
        return "emb:" + hashlib.sha256(f"{self.model_id}\0{text}".encode()).hexdigest()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(self._key(t))
            if hit is None:
                missing.append(i)
            else:
                out[i] = json.loads(hit)
        if missing:
            for i, v in zip(missing, self.inner.embed([texts[i] for i in missing]), strict=True):
                out[i] = v
                self.cache.set(self._key(texts[i]), json.dumps(v), self.ttl)
        return out  # type: ignore[return-value]
