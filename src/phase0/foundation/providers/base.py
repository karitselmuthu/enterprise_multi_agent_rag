from __future__ import annotations

import time
from abc import ABC, abstractmethod

from ..models import InferenceRequest, InferenceResult


class ProviderError(RuntimeError):
    pass


class BaseProvider(ABC):
    name: str

    def __init__(self, default_model: str) -> None:
        self.default_model = default_model

    @abstractmethod
    def complete(self, request: InferenceRequest, model: str | None = None) -> InferenceResult:
        raise NotImplementedError

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text.split()))

    @staticmethod
    def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
        cost_per_1k = {"local": 0.0001, "bedrock": 0.02, "vertex": 0.015}
        return ((input_tokens + output_tokens) / 1000) * cost_per_1k.get(provider, 0.02)

    def _build_result(self, text: str, request: InferenceRequest, model: str, latency_start: float) -> InferenceResult:
        latency_ms = (time.perf_counter() - latency_start) * 1000
        input_tokens = self.estimate_tokens(request.prompt)
        output_tokens = self.estimate_tokens(text)
        return InferenceResult(
            text=text,
            provider=self.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=self.estimate_cost(self.name, input_tokens, output_tokens),
        )

