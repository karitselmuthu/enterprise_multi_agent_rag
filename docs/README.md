# Enterprise Multi-Agent RAG

This project is adapted for **8 GB laptops** where local inference memory is limited to ~4 GB.

## Project layout

`code-rag-platform/` now follows app/package separation:

- `apps/code_rag_cli/main.py` - CLI entry (`--engine direct|langgraph`)
- `apps/code_rag_api/main.py` - FastAPI HTTP wrapper (`POST /orchestrate`, `GET /health`)
- `src/code_rag_platform/agents/` - routing + orchestration
- `src/code_rag_platform/core/` - guardrails, ingestion, retrieval, vector store
- `src/code_rag_platform/config/` - settings and thresholds

`src/phase0/` now follows phase-based separation:

- `foundation/` - model gateway, providers, policies, telemetry, core models
- `phase1/` - ingestion, qdrant, dependency graph, indexing pipeline
- `phase2/` - runtime flow primitives
- `phase3/` - retrieval security and audit controls
- `phase4/` - evaluation and cost tuning utilities
- `phase5/` - rollout and approval controls

## Current implementation status

- Phase 0: Model gateway, routing/fallback, budget/rate controls, cache, telemetry, regression checks
- Phase 1: Tree-sitter parsing (with fixed-size fallback for non-code files), Qdrant vector client wiring (or in-memory store when `LOCAL_MODE=true`), dependency graph persistence, secret scanning, metadata tagging, incremental planning
- Phase 2: runtime scaffolding with fast/standard/deep paths and conditional verification
- Phase 3: retrieval security, tool guardrails, immutable audit log
- Phase 4: retrieval/answer metrics and cost-aware tuning primitives
- Phase 5: human-approval gate and progressive rollout manager
- `code-rag-platform` orchestrator is now wired to live indexing + retrieval using the persistent dependency graph and Phase 1 pipeline

## 8 GB operating model

- Keep a single local model active at a time (1.5B–3B quantized)
- Route complex RCA/architecture/security work to Bedrock/Vertex
- Prefer remote Qdrant under storage constraints
- Keep sensitive and restricted content guarded before retrieval and cloud escalation

## Commands

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p "test_*.py"
PYTHONPATH=src python3 -m phase0.main --eval --dataset evals/golden_phase0.json
```
