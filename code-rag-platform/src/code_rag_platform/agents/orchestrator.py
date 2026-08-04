from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from code_rag_platform.agents.router import ExecutionRouter, RouteInput
from code_rag_platform.config.settings import AppSettings
from code_rag_platform.core.guardrails.dlp_scrubber import DLPScrubber
from code_rag_platform.core.retrieval.live_indexed_retriever import LiveIndexedRetriever


@dataclass
class OrchestratorState:
    user_query: str
    task_type: str
    repo: str = "local"
    source_path: str = ""
    source_content: str = ""
    owner_team: str = "platform"
    classification: str = "internal"
    acl_roles: list[str] = field(default_factory=lambda: ["dev"])
    risk_level: str = "low"
    confidence: float = 0.8
    evidence_coverage: float = 0.8
    route: dict[str, Any] = field(default_factory=dict)
    sanitized_query: str = ""
    response: str = ""
    verified: bool = False
    retrieval_hits: list[dict[str, Any]] = field(default_factory=list)
    dependency_context: list[str] = field(default_factory=list)


class PhaseOrchestrator:
    _retrieval_task_types = {"ownership_lookup", "code_search", "repo_metadata_lookup", "incident_rca"}

    def __init__(self, settings: AppSettings | None = None, retriever: LiveIndexedRetriever | None = None) -> None:
        self.settings = settings or AppSettings()
        self.router = ExecutionRouter(self.settings)
        self.scrubber = DLPScrubber()
        self.retriever = retriever

    def _get_retriever(self) -> LiveIndexedRetriever:
        if self.retriever is None:
            self.retriever = LiveIndexedRetriever(self.settings)
        return self.retriever

    def prepare(self, state: OrchestratorState) -> OrchestratorState:
        """Scrub, index, classify the execution path, and retrieve. Shared by run() and the LangGraph nodes."""
        if self.scrubber.has_high_risk_content(state.user_query):
            state.sanitized_query = self.scrubber.scrub(state.user_query)
        else:
            state.sanitized_query = state.user_query

        if state.source_path and state.source_content:
            chunks_indexed, dependencies_indexed = self._get_retriever().index_document(
                repo=state.repo,
                path=state.source_path,
                owner_team=state.owner_team,
                classification=state.classification,
                acl_roles=tuple(state.acl_roles),
                content=state.source_content,
            )
            index_stats = {"chunks_indexed": chunks_indexed, "dependencies_indexed": dependencies_indexed}
        else:
            index_stats = None

        decision = self.router.classify(
            RouteInput(
                task_type=state.task_type,
                risk_level=state.risk_level,
                confidence=state.confidence,
                evidence_coverage=state.evidence_coverage,
            )
        )
        state.route = {
            "execution_path": decision.execution_path,
            "escalate_to_cloud": decision.escalate_to_cloud,
            "needs_verification": decision.needs_verification,
            "local_model": self.settings.providers.local_model,
            "cloud_model": self.settings.providers.bedrock_model,
        }
        if index_stats is not None:
            state.route["index_stats"] = index_stats

        if state.task_type in self._retrieval_task_types and state.source_path:
            retrieval = self._get_retriever().retrieve(
                query=state.sanitized_query,
                repo=state.repo,
                module_path=state.source_path,
            )
            state.retrieval_hits = retrieval.hits
            state.dependency_context = retrieval.dependencies

        state.verified = decision.needs_verification
        return state

    def respond_fast(self, state: OrchestratorState) -> OrchestratorState:
        state.response = (
            f"Fast path result for: {state.sanitized_query[:180]} "
            f"(hits={len(state.retrieval_hits)}, deps={len(state.dependency_context)})"
        )
        return state

    def respond_deep(self, state: OrchestratorState) -> OrchestratorState:
        state.response = (
            f"Deep investigation prepared for cloud escalation: {state.sanitized_query[:180]} "
            f"(hits={len(state.retrieval_hits)}, deps={len(state.dependency_context)})"
        )
        return state

    def respond_standard(self, state: OrchestratorState) -> OrchestratorState:
        state.response = (
            f"Standard investigation prepared: {state.sanitized_query[:180]} "
            f"(hits={len(state.retrieval_hits)}, deps={len(state.dependency_context)})"
        )
        return state

    _responders: dict[str, str] = {"fast": "respond_fast", "deep": "respond_deep", "standard": "respond_standard"}

    def run(self, state: OrchestratorState) -> OrchestratorState:
        state = self.prepare(state)
        responder = getattr(self, self._responders[state.route["execution_path"]])
        return responder(state)


def build_langgraph_state_machine(orchestrator: PhaseOrchestrator | None = None) -> Any:
    """Real multi-path routing graph: prepare -> classify -> {fast,standard,deep} -> END."""
    from langgraph.graph import END, StateGraph  # type: ignore

    orch = orchestrator or PhaseOrchestrator()
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("prepare", orch.prepare)
    workflow.add_node("fast", orch.respond_fast)
    workflow.add_node("standard", orch.respond_standard)
    workflow.add_node("deep", orch.respond_deep)
    workflow.set_entry_point("prepare")
    workflow.add_conditional_edges(
        "prepare",
        lambda state: state.route["execution_path"],
        {"fast": "fast", "standard": "standard", "deep": "deep"},
    )
    workflow.add_edge("fast", END)
    workflow.add_edge("standard", END)
    workflow.add_edge("deep", END)
    return workflow.compile()
