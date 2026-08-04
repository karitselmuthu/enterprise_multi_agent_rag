from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .ingestion import IndexedChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key
        self._client_factory = client_factory or self._default_client_factory
        self._client = self._client_factory(url=self.url, api_key=self.api_key)

    @staticmethod
    def _default_client_factory(url: str, api_key: str | None) -> Any:
        from qdrant_client import QdrantClient  # type: ignore

        return QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.http import models as rest  # type: ignore

        if self._client.collection_exists(self.collection_name):
            return
        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            on_disk_payload=True,
        )

    def upsert_chunks(self, chunks: list[IndexedChunk], dense_vectors: list[list[float]]) -> None:
        if len(chunks) != len(dense_vectors):
            raise ValueError("chunks and dense_vectors lengths must match")
        from qdrant_client.http import models as rest  # type: ignore

        points: list[Any] = []
        for chunk, vector in zip(chunks, dense_vectors, strict=True):
            payload = {
                "text": chunk.text,
                "source_path": chunk.metadata.source_path,
                "repo": chunk.metadata.repo,
                "owner_team": chunk.metadata.owner_team,
                "classification": chunk.metadata.classification,
                "acl_roles": list(chunk.metadata.acl_roles),
                "content_hash": chunk.metadata.content_hash,
            }
            points.append(rest.PointStruct(id=chunk.chunk_id, vector=vector, payload=payload))
        self._client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query_vector: list[float], limit: int = 5) -> list[RetrievedChunk]:
        hits = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk_id=str(hit.id), score=float(hit.score), payload=dict(hit.payload or {}))
            for hit in hits
        ]
