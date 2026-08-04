from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HardwareProfile:
    system_ram_gb: int = int(os.getenv("SYSTEM_RAM_GB", "8"))
    available_model_ram_gb: float = float(os.getenv("AVAILABLE_MODEL_RAM_GB", "4.0"))
    max_concurrent_models: int = int(os.getenv("MAX_CONCURRENT_MODELS", "1"))
    local_storage_budget_gb: int = int(os.getenv("LOCAL_STORAGE_BUDGET_GB", "256"))


@dataclass(frozen=True)
class RoutingThresholds:
    fast_path_confidence: float = float(os.getenv("FAST_PATH_CONFIDENCE", "0.85"))
    verify_if_confidence_below: float = float(os.getenv("VERIFY_IF_CONFIDENCE_BELOW", "0.75"))
    verify_if_evidence_below: float = float(os.getenv("VERIFY_IF_EVIDENCE_BELOW", "0.70"))


@dataclass(frozen=True)
class ProviderSettings:
    local_mode: bool = os.getenv("LOCAL_MODE", "false").lower() == "true"
    prefer_remote_qdrant: bool = os.getenv("PREFER_REMOTE_QDRANT", "true").lower() == "true"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "code_chunks")
    dependency_graph_db_path: str = os.getenv("DEPENDENCY_GRAPH_DB_PATH", "code-rag-platform/data/dependency_graph.sqlite")
    local_model: str = os.getenv("LOCAL_MODEL", "qwen2.5-coder-1.5b-instruct-q4")
    bedrock_model: str = os.getenv("BEDROCK_MODEL", "anthropic.claude-3-5-sonnet")
    vertex_model: str = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")


@dataclass(frozen=True)
class AppSettings:
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    routing: RoutingThresholds = field(default_factory=RoutingThresholds)
    providers: ProviderSettings = field(default_factory=ProviderSettings)
