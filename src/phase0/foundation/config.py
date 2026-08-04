from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderConfig:
    enabled: bool
    default_model: str


@dataclass(frozen=True)
class BudgetConfig:
    user_daily_usd: float = 3.0
    team_daily_usd: float = 100.0


@dataclass(frozen=True)
class RateLimitConfig:
    requests_per_minute: int = 60
    burst: int = 10


@dataclass(frozen=True)
class CacheConfig:
    prompt_ttl_seconds: int = 1800
    semantic_ttl_seconds: int = 900
    semantic_similarity_threshold: float = 0.86


@dataclass(frozen=True)
class RoutingConfig:
    premium_task_types: tuple[str, ...] = (
        "incident_rca",
        "architecture_review",
        "security_review",
        "multi_service_dependency",
    )
    risk_escalation_levels: tuple[str, ...] = ("high", "critical")
    local_fallback_model: str = "qwen2.5-coder-1.5b-instruct-q4"
    cloud_required_task_types: tuple[str, ...] = (
        "incident_rca",
        "architecture_review",
        "security_review",
        "multi_service_dependency",
        "deep_code_fix",
    )


@dataclass(frozen=True)
class HardwareConfig:
    system_ram_gb: int = 8
    available_model_ram_gb: float = 4.0
    max_concurrent_models: int = 1
    local_storage_budget_gb: int = 256
    prefer_remote_qdrant: bool = True
    low_memory_local_models: tuple[str, ...] = (
        "qwen2.5-coder-1.5b-instruct-q4",
        "qwen2.5-3b-instruct-q4",
    )


@dataclass(frozen=True)
class GatewayConfig:
    local: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(True, "qwen2.5-3b-instruct-q4")
    )
    bedrock: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(True, "anthropic.claude-3-5-sonnet")
    )
    vertex: ProviderConfig = field(
        default_factory=lambda: ProviderConfig(True, "gemini-2.5-pro")
    )
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
