from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


class LowFootprintQdrantManager:
    def __init__(self, url: str, collection: str, api_key: str | None = None) -> None:
        self.url = url
        self.collection = collection
        self.api_key = api_key
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient  # type: ignore

            self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=8.0)
        return self._client

    def ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.http import models as rest  # type: ignore

        client = self._get_client()
        if client.collection_exists(self.collection):
            return
        client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            on_disk_payload=True,
        )

    def upsert(self, chunk_vectors: list[tuple[str, list[float], dict[str, Any]]]) -> None:
        from qdrant_client.http import models as rest  # type: ignore

        client = self._get_client()
        points = [
            rest.PointStruct(id=chunk_id, vector=vector, payload=payload)
            for chunk_id, vector, payload in chunk_vectors
        ]
        client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, vector: list[float], limit: int = 5) -> list[RetrievalHit]:
        client = self._get_client()
        results = client.search(
            collection_name=self.collection,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievalHit(
                chunk_id=str(item.id),
                score=float(item.score),
                payload=dict(item.payload or {}),
            )
            for item in results
        ]

