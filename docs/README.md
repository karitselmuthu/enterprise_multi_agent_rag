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
- `code-rag-platform` orchestrator is wired end-to-end: live indexing + retrieval (Phase 1 pipeline + dependency graph) feeds generation through the Phase 0 `ModelGateway`, so every request actually routes to local/Bedrock/Vertex, not just a label

## 8 GB operating model

- Keep a single local model active at a time (1.5B–3B quantized)
- Route complex RCA/architecture/security work to Bedrock/Vertex
- Prefer remote Qdrant under storage constraints
- Keep sensitive and restricted content guarded before retrieval and cloud escalation

## Fork & test guide

### Prerequisites

- **Python 3.10 or 3.11.** `tree-sitter-languages` (AST chunking) ships no wheels for 3.13/3.14 — pick an
  older interpreter for the venv (e.g. `python3.10 -m venv .venv`), not whatever `python3` resolves to by default.
- No cloud credentials are required to run or test this app today. Every provider (`local`, `bedrock`, `vertex`)
  is a stub that returns `[provider:model] prompt...` — the value being tested is **routing**, not live inference.
  See "Going live" below for what changes when you're ready to call real APIs.

### 1. Local

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -e ".[phase1]"
pip install -r code-rag-platform/requirements.txt

pytest tests/ -v                                    # 24 unit tests
python3 -m phase0.main --eval                        # 6 golden-case regression (evals/golden_phase0.json)

# Orchestrator CLI, offline (LOCAL_MODE swaps Qdrant for an in-memory store)
cd code-rag-platform
LOCAL_MODE=true python main.py --mode orchestrate --query "who owns checkout" \
  --task-type ownership_lookup --source-path svc.py --source-content "import os"

# FastAPI wrapper
cd ..
LOCAL_MODE=true PYTHONPATH=code-rag-platform/src:code-rag-platform/apps:src \
  uvicorn code_rag_api.main:app --reload --port 8000
# then: curl http://127.0.0.1:8000/health
```

Confirms: DLP scrubbing, retrieval + dependency graph, LangGraph multi-path routing, and that low-risk/
non-premium tasks select the small **8GB-safe local model** (`qwen2.5-coder-1.5b-instruct-q4`, see
`src/phase0/foundation/config.py::HardwareConfig.low_memory_local_models`).

### 2. "AWS" (Bedrock routing)

Any task in `GatewayConfig.routing.premium_task_types` / `cloud_required_task_types`
(`security_review`, `architecture_review`, `incident_rca`, `multi_service_dependency`, `deep_code_fix`) or any
`risk_level` of `high`/`critical` routes to `bedrock`:

```bash
LOCAL_MODE=true python code-rag-platform/main.py --mode orchestrate \
  --query "checkout is down" --task-type incident_rca --risk-level high --confidence 0.5 --evidence 0.4
# route.execution_path == "deep", route.model_provider == "bedrock"
```

Or the eval case that pins this exactly: `deep_code_fix_forced_cloud_on_8gb_laptop` in
`evals/golden_phase0.json` — proves cloud is forced purely by the 8GB RAM ceiling, independent of task risk.

**Going live:** `BedrockProvider.complete()` in `src/phase0/foundation/providers/bedrock.py` is where a real
call belongs (`boto3` `bedrock-runtime.invoke_model`, region from `BEDROCK_REGION`). Note `.env.example`'s
`BEDROCK_REGION` isn't read by anything yet — `GatewayConfig` is a hardcoded dataclass — so wiring a real call
also means making that field `os.getenv`-driven, the same way `code_rag_platform/config/settings.py` already
does it for Qdrant.

### 3. "GCP" (Vertex routing)

Vertex is the fallback cloud provider (used when Bedrock is disabled) and can be forced directly:

```bash
python3 -c "
from phase0.foundation.config import GatewayConfig, ProviderConfig
from phase0.foundation.gateway import ModelGateway
from phase0.foundation.models import InferenceRequest
config = GatewayConfig(bedrock=ProviderConfig(False, 'anthropic.claude-3-5-sonnet'))
gw = ModelGateway(config)
r = gw.infer(InferenceRequest(user_id='u', agent_name='a', prompt='security review of auth', task_type='security_review'))
print(r.provider, r.model)  # vertex gemini-2.5-pro
"
```

**Going live:** same pattern as Bedrock — `VertexProvider.complete()` in
`src/phase0/foundation/providers/vertex.py` is where a real `vertexai` SDK call belongs, reading
`VERTEX_PROJECT_ID`/`VERTEX_REGION` (also currently unread placeholders in `.env.example`).

### Commands reference

```bash
pytest tests/ -v                                                       # full suite
python3 -m phase0.main --prompt "..." --task-type ownership_lookup     # gateway CLI
python3 -m phase0.main --eval                                          # golden regression
python3 code-rag-platform/main.py --mode orchestrate --query "..."     # orchestrator CLI
python3 code-rag-platform/main.py --mode orchestrate --engine langgraph --query "..."
uvicorn code_rag_api.main:app --port 8000                              # FastAPI (see PYTHONPATH above)
```
