from __future__ import annotations

import threading
import time
from typing import Callable


class RateLimiter:
    """Per-key token bucket (in-process; use a gateway for limits shared across replicas)."""

    def __init__(self, per_minute: int, burst: int | None = None, clock: Callable[[], float] = time.monotonic):
        self.rate = per_minute / 60.0
        self.capacity = float(burst or max(1, per_minute))
        self.clock = clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> float:
        """0.0 if allowed, else the seconds to wait before retrying."""
        with self._lock:
            now = self.clock()
            tokens, last = self._state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._state[key] = (tokens - 1.0, now)
                return 0.0
            self._state[key] = (tokens, now)
            return (1.0 - tokens) / self.rate
