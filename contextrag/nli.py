from __future__ import annotations

import re
from typing import Protocol, Sequence

from .text import content_tokens, stem, tokenize


class NLIModel(Protocol):
    def entailment(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """P(premise entails hypothesis) for each (premise, hypothesis) pair."""
        ...


_NEG = {"no", "not", "never", "none", "without", "cannot", "nunca", "sin", "tampoco", "nadie", "ningún", "ninguno"}
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


class LexicalEntailment:
    """Conservative offline stand-in for an NLI model.

    Score is the share of the hypothesis' content terms found in the premise; it drops to 0
    if the hypothesis introduces a number the premise lacks, or flips negation.
    It over-rejects paraphrase by design (it can't be fooled into approving), but it cannot
    detect subtle contradictions — use ``DebertaNLI`` in production.
    """

    def entailment(self, pairs):
        return [self._one(p, h) for p, h in pairs]

    @staticmethod
    def _one(premise: str, hyp: str) -> float:
        h = {stem(t) for t in content_tokens(hyp)}
        if not h:
            return 0.0
        p = {stem(t) for t in content_tokens(premise)}
        if set(_NUM.findall(hyp)) - set(_NUM.findall(premise)):
            return 0.0
        if bool(_NEG & set(tokenize(hyp))) != bool(_NEG & set(tokenize(premise))):
            return 0.0
        return len(h & p) / len(h)


class DebertaNLI:
    """DeBERTa-v3 NLI cross-encoder (needs ``transformers`` + ``torch``)."""

    def __init__(self, model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model)
        self._m = AutoModelForSequenceClassification.from_pretrained(model).eval()
        labels = {k.lower(): v for k, v in self._m.config.label2id.items()}
        self._ent = labels["entailment"]
        self.batch_size = batch_size

    def entailment(self, pairs):
        out: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            enc = self._tok([p for p, _ in batch], [h for _, h in batch], truncation=True, max_length=512,
                            padding=True, return_tensors="pt")
            with self._torch.no_grad():
                probs = self._m(**enc).logits.softmax(-1)[:, self._ent]
            out.extend(probs.tolist())
        return out
