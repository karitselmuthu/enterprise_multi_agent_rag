from __future__ import annotations

import argparse
import json

from code_rag_platform.agents.orchestrator import OrchestratorState, PhaseOrchestrator
from code_rag_platform.core.guardrails.dlp_scrubber import DLPScrubber
from code_rag_platform.core.ingestion.chunker import TreeSitterChunker, tag_chunks


def run() -> None:
    parser = argparse.ArgumentParser(description="Code RAG Platform CLI")
    parser.add_argument("--mode", choices=["orchestrate", "scrub", "chunk"], default="orchestrate")
    parser.add_argument("--query", default="Find owner of checkout service")
    parser.add_argument("--task-type", default="ownership_lookup")
    parser.add_argument("--risk-level", default="low")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--evidence", type=float, default=0.80)
    parser.add_argument("--path", default="sample.py")
    parser.add_argument("--repo", default="local")
    parser.add_argument("--source-path", default="")
    parser.add_argument("--source-content", default="")
    parser.add_argument("--owner-team", default="platform")
    parser.add_argument("--classification", default="internal")
    parser.add_argument("--acl-roles", default="dev")
    parser.add_argument("--engine", choices=["direct", "langgraph"], default="direct")
    args = parser.parse_args()

    if args.mode == "scrub":
        scrubber = DLPScrubber()
        print(scrubber.scrub(args.query))
        return

    if args.mode == "chunk":
        chunker = TreeSitterChunker()
        chunks = chunker.chunk(args.path, args.query)
        tagged = tag_chunks(
            repo="local",
            path=args.path,
            owner_team="platform",
            classification="internal",
            acl_roles=["dev"],
            chunks=chunks,
        )
        print(json.dumps([item.__dict__ for item in tagged], indent=2))
        return

    orchestrator = PhaseOrchestrator()
    initial_state = OrchestratorState(
        user_query=args.query,
        task_type=args.task_type,
        repo=args.repo,
        source_path=args.source_path,
        source_content=args.source_content,
        owner_team=args.owner_team,
        classification=args.classification,
        acl_roles=[role.strip() for role in args.acl_roles.split(",") if role.strip()],
        risk_level=args.risk_level,
        confidence=args.confidence,
        evidence_coverage=args.evidence,
    )
    if args.engine == "langgraph":
        from code_rag_platform.agents.orchestrator import build_langgraph_state_machine

        state = build_langgraph_state_machine(orchestrator).invoke(initial_state)
        print(json.dumps(state if isinstance(state, dict) else state.__dict__, indent=2, default=str))
        return

    state = orchestrator.run(initial_state)
    print(json.dumps(state.__dict__, indent=2))


if __name__ == "__main__":
    run()
