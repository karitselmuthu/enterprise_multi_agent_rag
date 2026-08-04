# Phase Build Plan (8 GB Profile)

## Phase 1: Knowledge Ingestion and Dependency Intelligence
- Tree-sitter parser-backed structural chunking
- Secret scan before indexing
- Classification + ACL metadata attachment
- Incremental targets from changed files
- Remote vector index preference when storage is constrained
- Qdrant collection management, upsert, and similarity search wiring
- SQLite-backed dependency graph nodes and edges persistence

## Phase 2: Hybrid Multi-Agent Runtime
- Fast path for simple lookups
- Standard path for common tasks
- Deep path for high complexity with cloud forcing
- Verification based on risk/confidence/evidence coverage

## Phase 3: Retrieval Security and Governance
- Entitlement and classification filtering before retrieval
- Tool allowlist and unsafe parameter blocking
- Hash-chained immutable audit records

## Phase 4: Evaluation and Cost Optimisation
- Retrieval metrics (precision/recall/MRR/citation quality)
- Answer metrics (groundedness/completeness/latency/cost)
- Cost-aware downgrade signal at acceptable quality

## Phase 5: Developer Experience and Rollout
- Human approval for risky production actions
- Team rollout stages (pilot → department → enterprise)
