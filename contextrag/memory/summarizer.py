"""Turn-to-memory summarization.

The extractive summarizer keeps sentences the *user* wrote that state a decision, a preference or a task. It copies
them verbatim, so nothing is paraphrased or invented, and marks them ``user_stated`` with confidence 1.0. Assistant
turns are never promoted: what the model said is its own output, not something the user established.

An LLM-based summarizer would have to produce ``model_inferred`` memories with confidence below 1.0 and must go
through the same validation (source turns inside the job range). It is not implemented here.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from ..core.text import split_sentences
from .store import MemoryDraft, Turn

_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("decision", re.compile(r"\b(decidimos|decid[ií]|acordamos|decisi[oó]n|decision|we decided|we agreed)\b", re.I)),
    ("preference", re.compile(r"\b(prefiero|prefiere|no quiero|me gusta|no me gusta|i prefer|i like)\b", re.I)),
    ("task", re.compile(r"\b(tarea|pendiente|debo|tengo que|necesito|por hacer|action item|need to)\b", re.I)),
)


class Summarizer(Protocol):
    def summarize(self, turns: Sequence[Turn]) -> list[MemoryDraft]: ...


class ExtractiveSummarizer:
    def __init__(self, max_chars: int = 500):
        if max_chars < 1:
            raise ValueError("max_chars must be >= 1")
        self.max_chars = max_chars

    def summarize(self, turns: Sequence[Turn]) -> list[MemoryDraft]:
        drafts: list[MemoryDraft] = []
        seen: set[str] = set()
        for turn in turns:
            if turn.role != "user":
                continue
            for sentence, _, _ in split_sentences(turn.content):
                text = sentence.strip()[: self.max_chars]
                kind = next((k for k, pattern in _KINDS if pattern.search(text)), None)
                if kind is None or text in seen:
                    continue
                seen.add(text)
                drafts.append(MemoryDraft(kind, text, "user_stated", 1.0, turn.seq))
        return drafts
