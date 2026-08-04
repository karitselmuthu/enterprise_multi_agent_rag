from __future__ import annotations

from dataclasses import dataclass

from ..foundation.gateway import ModelGateway
from ..foundation.models import InferenceRequest, InferenceResult


@dataclass(frozen=True)
class RuntimeRequest:
    user_id: str
    agent_name: str
    task_type: str
    prompt: str
    risk_level: str
    confidence: float
    evidence_coverage: float


@dataclass(frozen=True)
class RuntimeResponse:
    execution_path: str
    verified: bool
    result: InferenceResult


class MultiAgentRuntime:
    def __init__(self, gateway: ModelGateway) -> None:
        self.gateway = gateway

    def _execution_path(self, request: RuntimeRequest) -> str:
        if request.task_type in {"ownership_lookup", "code_search"} and request.risk_level == "low":
            return "fast"
        if request.task_type in {"incident_rca", "architecture_review", "security_review"}:
            return "deep"
        return "standard"

    def _needs_verification(self, request: RuntimeRequest) -> bool:
        if request.risk_level in {"high", "critical"}:
            return True
        if request.confidence < 0.75 or request.evidence_coverage < 0.7:
            return True
        return False

    def execute(self, request: RuntimeRequest) -> RuntimeResponse:
        path = self._execution_path(request)
        force_cloud = path == "deep"
        result = self.gateway.infer(
            InferenceRequest(
                user_id=request.user_id,
                agent_name=request.agent_name,
                prompt=request.prompt,
                task_type=request.task_type,
                risk_level=request.risk_level,
                metadata={"force_cloud": force_cloud},
            )
        )
        return RuntimeResponse(
            execution_path=path,
            verified=self._needs_verification(request),
            result=result,
        )
