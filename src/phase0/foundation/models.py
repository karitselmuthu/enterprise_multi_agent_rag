from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceRequest:
    user_id: str
    agent_name: str
    prompt: str
    task_type: str
    risk_level: str = "low"
    max_tokens: int = 800
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool = False
    cache_type: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    provider: str
    model: str
    fallbacks: list[tuple[str, str]]

