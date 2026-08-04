from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .dependency_graph import DependencyGraphBuilder, DependencyGraphStore
from .ingestion import ChunkIndexer, IndexedChunk
from .qdrant import QdrantVectorStore


EmbeddingFn = Callable[[str], list[float]]


@dataclass(frozen=True)
class IndexedDocumentResult:
    chunks_indexed: int
    dependencies_indexed: int


class Phase1IndexerPipeline:
    def __init__(
        self,
        chunk_indexer: ChunkIndexer,
        vector_store: QdrantVectorStore,
        graph_store: DependencyGraphStore,
        graph_builder: DependencyGraphBuilder | None = None,
    ) -> None:
        self.chunk_indexer = chunk_indexer
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.graph_builder = graph_builder or DependencyGraphBuilder()

    def index_document(
        self,
        repo: str,
        path: str,
        owner_team: str,
        classification: str,
        acl_roles: tuple[str, ...],
        content: str,
        embedding_fn: EmbeddingFn,
    ) -> IndexedDocumentResult:
        module_id = f"{repo}:{path}"
        self.graph_store.upsert_node(
            node_id=module_id,
            kind="module",
            repo=repo,
            owner_team=owner_team,
            metadata={"path": path, "classification": classification},
        )
        chunks = self.chunk_indexer.build_chunks(
            repo=repo,
            path=path,
            owner_team=owner_team,
            classification=classification,
            acl_roles=acl_roles,
            content=content,
        )
        dense_vectors = [embedding_fn(chunk.text) for chunk in chunks]
        if dense_vectors:
            self.vector_store.ensure_collection(vector_size=len(dense_vectors[0]))
            self.vector_store.upsert_chunks(chunks=chunks, dense_vectors=dense_vectors)

        edges = self.graph_builder.extract_edges(
            repo=repo,
            module_id=module_id,
            path=path,
            content=content,
        )
        for edge in edges:
            self.graph_store.upsert_node(
                node_id=edge.target,
                kind="dependency",
                repo=repo,
                owner_team=owner_team,
                metadata={"discovered_from": path},
            )
            self.graph_store.upsert_edge(edge)
        return IndexedDocumentResult(chunks_indexed=len(chunks), dependencies_indexed=len(edges))
