from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass

from .config import CacheConfig
from .models import InferenceResult


@dataclass
class _CacheEntry:
    value: InferenceResult
    expires_at: float
    tokens: set[str]


class GatewayCache:
    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._prompt_cache: dict[str, _CacheEntry] = {}
        self._semantic_cache: dict[str, _CacheEntry] = {}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union

    def get_prompt(self, prompt: str) -> InferenceResult | None:
        now = time.time()
        key = self._hash(prompt)
        entry = self._prompt_cache.get(key)
        if entry and entry.expires_at > now:
            return entry.value
        return None

    def put_prompt(self, prompt: str, value: InferenceResult) -> None:
        self._prompt_cache[self._hash(prompt)] = _CacheEntry(
            value=value,
            expires_at=time.time() + self._config.prompt_ttl_seconds,
            tokens=self._tokenize(prompt),
        )

    def get_semantic(self, prompt: str) -> InferenceResult | None:
        now = time.time()
        tokens = self._tokenize(prompt)
        best: _CacheEntry | None = None
        best_score = 0.0
        for entry in self._semantic_cache.values():
            if entry.expires_at <= now:
                continue
            score = self._jaccard(tokens, entry.tokens)
            if score > best_score:
                best = entry
                best_score = score
        if best and best_score >= self._config.semantic_similarity_threshold:
            return best.value
        return None

    def put_semantic(self, prompt: str, value: InferenceResult) -> None:
        self._semantic_cache[self._hash(prompt)] = _CacheEntry(
            value=value,
            expires_at=time.time() + self._config.semantic_ttl_seconds,
            tokens=self._tokenize(prompt),
        )

