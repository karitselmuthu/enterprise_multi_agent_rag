from __future__ import annotations

from dataclasses import replace

from .cache import GatewayCache
from .config import GatewayConfig
from .models import InferenceRequest, InferenceResult, RouteDecision
from .policies import BudgetTracker, RateLimiter
from .providers import BedrockProvider, LocalProvider, VertexProvider
from .providers.base import BaseProvider, ProviderError
from .telemetry import Telemetry


class ModelGateway:
    def __init__(self, config: GatewayConfig | None = None) -> None:
        self.config = config or GatewayConfig()
        self.telemetry = Telemetry()
        self.cache = GatewayCache(self.config.cache)
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self.budget_tracker = BudgetTracker(self.config.budget)
        self.providers: dict[str, BaseProvider] = {}
        if self.config.local.enabled:
            self.providers["local"] = LocalProvider(self.config.local.default_model)
        if self.config.bedrock.enabled:
            self.providers["bedrock"] = BedrockProvider(self.config.bedrock.default_model)
        if self.config.vertex.enabled:
            self.providers["vertex"] = VertexProvider(self.config.vertex.default_model)

    def _route(self, request: InferenceRequest) -> RouteDecision:
        wants_premium = (
            request.task_type in self.config.routing.premium_task_types
            or request.risk_level in self.config.routing.risk_escalation_levels
            or request.metadata.get("force_cloud", False)
        )
        low_memory_mode = self.config.hardware.available_model_ram_gb <= 4.5
        cloud_required = request.task_type in self.config.routing.cloud_required_task_types
        if low_memory_mode and cloud_required:
            if "bedrock" in self.providers:
                return RouteDecision(
                    "bedrock",
                    self.config.bedrock.default_model,
                    [("vertex", self.config.vertex.default_model), ("local", self.config.routing.local_fallback_model)],
                )
            if "vertex" in self.providers:
                return RouteDecision(
                    "vertex",
                    self.config.vertex.default_model,
                    [("bedrock", self.config.bedrock.default_model), ("local", self.config.routing.local_fallback_model)],
                )
        if wants_premium and "bedrock" in self.providers:
            fallbacks = [("vertex", self.config.vertex.default_model), ("local", self.config.routing.local_fallback_model)]
            return RouteDecision("bedrock", self.config.bedrock.default_model, fallbacks)
        if wants_premium and "vertex" in self.providers:
            fallbacks = [("bedrock", self.config.bedrock.default_model), ("local", self.config.routing.local_fallback_model)]
            return RouteDecision("vertex", self.config.vertex.default_model, fallbacks)
        fallbacks: list[tuple[str, str]] = []
        if "bedrock" in self.providers:
            fallbacks.append(("bedrock", self.config.bedrock.default_model))
        if "vertex" in self.providers:
            fallbacks.append(("vertex", self.config.vertex.default_model))
        preferred_local = self.config.local.default_model
        if low_memory_mode:
            preferred_local = self.config.hardware.low_memory_local_models[0]
        return RouteDecision("local", preferred_local, fallbacks)

    def _invoke_provider(self, provider_name: str, request: InferenceRequest, model: str) -> InferenceResult:
        provider = self.providers.get(provider_name)
        if not provider:
            raise ProviderError(f"Provider {provider_name} not available")
        return provider.complete(request, model=model)

    def _cached_result(self, result: InferenceResult, cache_type: str) -> InferenceResult:
        return replace(result, cached=True, cache_type=cache_type)

    def infer(self, request: InferenceRequest) -> InferenceResult:
        with self.telemetry.span("model_gateway.infer", agent_name=request.agent_name, task_type=request.task_type):
            self.telemetry.record_request()
            self.rate_limiter.assert_allowed(key=f"{request.user_id}:{request.agent_name}")

            prompt_cache_hit = self.cache.get_prompt(request.prompt)
            if prompt_cache_hit:
                self.telemetry.record_cache_hit("prompt")
                return self._cached_result(prompt_cache_hit, "prompt")

            semantic_cache_hit = self.cache.get_semantic(request.prompt)
            if semantic_cache_hit:
                self.telemetry.record_cache_hit("semantic")
                return self._cached_result(semantic_cache_hit, "semantic")

            self.telemetry.record_cache_miss()

            route = self._route(request)
            candidates = [(route.provider, route.model), *route.fallbacks]
            last_error: Exception | None = None
            for provider_name, model in candidates:
                self.telemetry.record_model_selection(provider_name, model)
                try:
                    self.budget_tracker.assert_within_budget(
                        user_id=request.user_id,
                        team_id=request.metadata.get("team_id", "default"),
                        expected_cost=0.01,
                    )
                    result = self._invoke_provider(provider_name, request, model)
                    self.budget_tracker.record_spend(
                        user_id=request.user_id,
                        team_id=request.metadata.get("team_id", "default"),
                        cost_usd=result.cost_usd,
                    )
                    self.telemetry.record_inference(
                        latency_ms=result.latency_ms,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cost_usd=result.cost_usd,
                    )
                    self.cache.put_prompt(request.prompt, result)
                    self.cache.put_semantic(request.prompt, result)
                    return result
                except Exception as exc:
                    last_error = exc
            if last_error:
                raise last_error
            raise ProviderError("No providers configured")
