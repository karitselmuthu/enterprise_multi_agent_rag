from __future__ import annotations

from dataclasses import dataclass

from code_rag_platform.config.settings import AppSettings


@dataclass(frozen=True)
class RouteInput:
    task_type: str
    risk_level: str
    confidence: float
    evidence_coverage: float


@dataclass(frozen=True)
class RouteDecision:
    execution_path: str
    escalate_to_cloud: bool
    needs_verification: bool


class ExecutionRouter:
    _deep_tasks = {"incident_rca", "architecture_review", "security_review", "multi_service_dependency"}
    _fast_tasks = {"ownership_lookup", "code_search", "repo_metadata_lookup"}

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or AppSettings()

    def classify(self, route_input: RouteInput) -> RouteDecision:
        low_memory = self.settings.hardware.available_model_ram_gb <= 4.5
        if route_input.task_type in self._deep_tasks:
            execution_path = "deep"
        elif (
            route_input.task_type in self._fast_tasks
            and route_input.confidence >= self.settings.routing.fast_path_confidence
            and route_input.risk_level == "low"
        ):
            execution_path = "fast"
        else:
            execution_path = "standard"

        escalate = execution_path == "deep" or (low_memory and route_input.risk_level in {"high", "critical"})
        verify = (
            route_input.risk_level in {"high", "critical"}
            or route_input.confidence < self.settings.routing.verify_if_confidence_below
            or route_input.evidence_coverage < self.settings.routing.verify_if_evidence_below
        )
        return RouteDecision(
            execution_path=execution_path,
            escalate_to_cloud=escalate,
            needs_verification=verify,
        )
