"""Best-effort redaction of secrets and common identifiers from log messages.

This is a second line of defence. The first is not to log content at all, which the code and its tests enforce for the
paths that handle documents, questions and answers. Redaction exists for what slips past that: an exception message from
a library, a connection string in an error, a value someone adds to a log line later.

What it recognises: Chilean RUT, email addresses, phone numbers, court case identifiers (``C-1234-2020``, ``Rol 123-2020``),
API keys and bearer tokens, and the password in a connection URL.

What it does NOT do, stated plainly: it is pattern matching, not understanding. It does not detect names, addresses,
free-text descriptions of a person's circumstances, or identifiers in formats it was not given, and it will redact an
occasional number that only looks like an identifier. It is not a data-loss-prevention system and it does not make
logging document content acceptable. It does not touch exception tracebacks printed by a handler, nor data stored in
the audit database.
"""
from __future__ import annotations

import logging
import os
import re

_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?<=://)([^:/\s@]+):[^@\s]+@"), r"\1:[REDACTED]@"),  # password inside a connection URL
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.I), "Bearer [REDACTED]"),
    (re.compile(r"\b(?:sk-ant-|sk-|crag_|ghp_|xox[bp]-|AKIA)[A-Za-z0-9_\-]{12,}"), "[SECRET]"),
    (re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"), "[RUT]"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    (re.compile(r"(?<![\w-])(?:\+?56[\s-]?)?9[\s-]?\d{4}[\s-]?\d{4}(?![\w-])"), "[PHONE]"),
    (re.compile(r"\+\d{1,3}[\s-]?\d{1,4}[\s-]?\d{3,4}[\s-]?\d{3,4}\b"), "[PHONE]"),
    (re.compile(r"\b(?:rol|causa|rit|ruc|expediente)\s*(?:n[°ºo.]*\s*)?[A-Za-z]?-?\d{1,7}-\d{4}\b", re.I), "[CASE]"),
    (re.compile(r"\b[CVTOLEF]-\d{1,6}-\d{4}\b"), "[CASE]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


_INSTALLED = "_contextrag_redacting"


def install_log_redaction() -> bool:
    """Redact every log record as it is created. Idempotent. Returns False if disabled by
    ``CONTEXTRAG_LOG_REDACTION=off`` (for debugging with synthetic data only)."""
    if os.getenv("CONTEXTRAG_LOG_REDACTION", "on").strip().lower() in ("off", "0", "false", "no"):
        return False
    current = logging.getLogRecordFactory()
    if getattr(current, _INSTALLED, False):
        return True

    def factory(*args, **kwargs):
        record = current(*args, **kwargs)
        try:
            message = record.getMessage()
        except Exception:  # a malformed format string is the caller's bug; do not lose the record
            return record
        clean = redact(message)
        if clean != message:
            record.msg, record.args = clean, None
        return record

    setattr(factory, _INSTALLED, True)
    logging.setLogRecordFactory(factory)
    return True
