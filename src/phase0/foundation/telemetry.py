from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger("phase0")

try:
    from opentelemetry import trace
except ImportError:  # pragma: no cover
    trace = None


@dataclass
class MetricSnapshot:
    requests: int = 0
    cache_hits_prompt: int = 0
    cache_hits_semantic: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    model_selection_count: dict[str, int] = field(default_factory=dict)


class Telemetry:
    def __init__(self) -> None:
        self.metrics = MetricSnapshot()
        self._tracer = trace.get_tracer("enterprise-rag-gateway") if trace else None

    @contextmanager
    def span(self, name: str, **attrs: str) -> Iterator[None]:
        if self._tracer:
            with self._tracer.start_as_current_span(name) as span:
                for key, value in attrs.items():
                    span.set_attribute(key, value)
                yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug("span=%s duration_ms=%.2f attrs=%s", name, elapsed, attrs)

    def record_request(self) -> None:
        self.metrics.requests += 1

    def record_cache_hit(self, cache_type: str) -> None:
        if cache_type == "prompt":
            self.metrics.cache_hits_prompt += 1
        elif cache_type == "semantic":
            self.metrics.cache_hits_semantic += 1

    def record_cache_miss(self) -> None:
        self.metrics.cache_misses += 1

    def record_model_selection(self, provider: str, model: str) -> None:
        key = f"{provider}:{model}"
        self.metrics.model_selection_count[key] = self.metrics.model_selection_count.get(key, 0) + 1

    def record_inference(self, latency_ms: float, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        self.metrics.total_latency_ms += latency_ms
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_cost_usd += cost_usd

