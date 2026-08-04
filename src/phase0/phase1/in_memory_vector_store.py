from __future__ import annotations

import math
from typing import Any

from .ingestion import IndexedChunk
from .qdrant import RetrievedChunk


class InMemoryVectorStore:
    """Drop-in QdrantVectorStore replacement for LOCAL_MODE: no server, no network."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._points: list[tuple[str, list[float], dict[str, Any]]] = []

    def ensure_collection(self, vector_size: int) -> None:
        return None

    def upsert_chunks(self, chunks: list[IndexedChunk], dense_vectors: list[list[float]]) -> None:
        if len(chunks) != len(dense_vectors):
            raise ValueError("chunks and dense_vectors lengths must match")
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
            self._points.append((chunk.chunk_id, vector, payload))

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def search(self, query_vector: list[float], limit: int = 5) -> list[RetrievedChunk]:
        scored = sorted(
            self._points,
            key=lambda point: self._cosine(query_vector, point[1]),
            reverse=True,
        )
        return [
            RetrievedChunk(chunk_id=chunk_id, score=self._cosine(query_vector, vector), payload=payload)
            for chunk_id, vector, payload in scored[:limit]
        ]
