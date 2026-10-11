"""Structured log events: one JSON object per line, so log shippers can index fields without parsing text."""
from __future__ import annotations

import json
import logging


def emit(logger: logging.Logger, level: int, event: str, *, exc_info: bool = False, **fields) -> None:
    """Log ``{"event": event, **fields}`` as one sorted JSON line (non-ASCII kept, unknown values stringified)."""
    logger.log(level, json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True, default=str),
               exc_info=exc_info)
