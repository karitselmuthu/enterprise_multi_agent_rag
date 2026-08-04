from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from code_rag_platform.agents.orchestrator import OrchestratorState, PhaseOrchestrator, build_langgraph_state_machine

app = FastAPI(title="Code RAG Platform API")
orchestrator = PhaseOrchestrator()
graph = build_langgraph_state_machine(orchestrator)


class OrchestrateRequest(BaseModel):
    query: str
    task_type: str = "ownership_lookup"
    repo: str = "local"
    source_path: str = ""
    source_content: str = ""
    owner_team: str = "platform"
    classification: str = "internal"
    acl_roles: list[str] = ["dev"]
    risk_level: str = "low"
    confidence: float = 0.85
    evidence_coverage: float = 0.80
    engine: Literal["direct", "langgraph"] = "direct"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "local_mode": str(orchestrator.settings.providers.local_mode)}


@app.post("/orchestrate")
def orchestrate(request: OrchestrateRequest) -> dict:
    state = OrchestratorState(
        user_query=request.query,
        task_type=request.task_type,
        repo=request.repo,
        source_path=request.source_path,
        source_content=request.source_content,
        owner_team=request.owner_team,
        classification=request.classification,
        acl_roles=request.acl_roles,
        risk_level=request.risk_level,
        confidence=request.confidence,
        evidence_coverage=request.evidence_coverage,
    )
    if request.engine == "langgraph":
        result = graph.invoke(state)
        return result if isinstance(result, dict) else asdict(result)
    return asdict(orchestrator.run(state))
