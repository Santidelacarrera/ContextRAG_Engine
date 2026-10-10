"""Key/value cache used for embeddings and query results (in-memory or Redis)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Protocol, Sequence


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


class CachedEmbedder:
    """Wraps an Embedder; vectors are cached by sha256(model, text)."""

    def __init__(self, inner, cache: Cache, name: str = "embedder", ttl: int | None = 7 * 86400):
        self.inner, self.cache, self.name, self.ttl = inner, cache, name, ttl
        self.dim = inner.dim

    def _key(self, text: str) -> str:
        return "emb:" + hashlib.sha256(f"{self.name}\0{text}".encode()).hexdigest()

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
            for i, v in zip(missing, self.inner.embed([texts[i] for i in missing])):
                out[i] = v
                self.cache.set(self._key(texts[i]), json.dumps(v), self.ttl)
        return out  # type: ignore[return-value]
