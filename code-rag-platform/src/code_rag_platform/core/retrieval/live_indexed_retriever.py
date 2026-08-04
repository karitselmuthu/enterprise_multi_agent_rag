from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from code_rag_platform.config.settings import AppSettings


def _ensure_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    src_path = repo_root / "src"
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


@dataclass(frozen=True)
class LiveRetrievalResult:
    indexed_chunks: int
    indexed_dependencies: int
    hits: list[dict[str, Any]]
    dependencies: list[str]


class LiveIndexedRetriever:
    def __init__(self, settings: AppSettings | None = None, pipeline: Any | None = None, graph_store: Any | None = None) -> None:
        self.settings = settings or AppSettings()
        if pipeline is not None and graph_store is not None:
            self.pipeline = pipeline
            self.graph_store = graph_store
            return
        _ensure_src_path()
        from phase0.phase1.dependency_graph import DependencyGraphStore
        from phase0.phase1.ingestion import ChunkIndexer
        from phase0.phase1.pipeline import Phase1IndexerPipeline

        db_path = Path(self.settings.providers.dependency_graph_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph_store = DependencyGraphStore(db_path)
        if self.settings.providers.local_mode:
            from phase0.phase1.in_memory_vector_store import InMemoryVectorStore

            vector_store: Any = InMemoryVectorStore()
        else:
            from phase0.phase1.qdrant import QdrantVectorStore

            vector_store = QdrantVectorStore(
                url=self.settings.providers.qdrant_url,
                collection_name=self.settings.providers.qdrant_collection,
                api_key=self.settings.providers.qdrant_api_key,
            )
        self.pipeline = Phase1IndexerPipeline(
            chunk_indexer=ChunkIndexer(),
            vector_store=vector_store,
            graph_store=self.graph_store,
        )

    @staticmethod
    def embed(text: str, dimensions: int = 16) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [digest[i] / 255.0 for i in range(dimensions)]
        return values

    def index_document(
        self,
        repo: str,
        path: str,
        owner_team: str,
        classification: str,
        acl_roles: tuple[str, ...],
        content: str,
    ) -> tuple[int, int]:
        indexed = self.pipeline.index_document(
            repo=repo,
            path=path,
            owner_team=owner_team,
            classification=classification,
            acl_roles=acl_roles,
            content=content,
            embedding_fn=self.embed,
        )
        return indexed.chunks_indexed, indexed.dependencies_indexed

    def retrieve(self, query: str, repo: str, module_path: str, limit: int = 5) -> LiveRetrievalResult:
        query_vector = self.embed(query)
        hits = self.pipeline.vector_store.search(query_vector=query_vector, limit=limit)
        module_id = f"{repo}:{module_path}"
        dependencies = [edge.target for edge in self.graph_store.downstream(module_id)]
        return LiveRetrievalResult(
            indexed_chunks=0,
            indexed_dependencies=0,
            hits=[{"chunk_id": hit.chunk_id, "score": hit.score, "payload": hit.payload} for hit in hits],
            dependencies=dependencies,
        )
