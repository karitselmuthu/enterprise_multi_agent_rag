from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from code_rag_platform._pathutil import ensure_src_path


@dataclass(frozen=True)
class TaggedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | list[str]]


class TreeSitterChunker:
    """Thin (path, content) adapter over phase0's ASTChunker — one AST-chunking implementation, not two."""

    def __init__(self, parser_resolver: Callable[[str], Any] | None = None) -> None:
        ensure_src_path()
        from phase0.phase1.ingestion import ASTChunker

        self._ast_chunker = ASTChunker(parser_resolver=parser_resolver)

    def chunk(self, path: str, content: str, chunk_size: int = 1200) -> list[str]:
        return self._ast_chunker.chunk(content, path=path, chunk_size=chunk_size)


def tag_chunks(
    repo: str,
    path: str,
    owner_team: str,
    classification: str,
    acl_roles: list[str],
    chunks: list[str],
) -> list[TaggedChunk]:
    out: list[TaggedChunk] = []
    for idx, chunk_text in enumerate(chunks):
        out.append(
            TaggedChunk(
                chunk_id=f"{repo}:{path}:{idx}",
                text=chunk_text,
                metadata={
                    "repo": repo,
                    "source_path": path,
                    "owner_team": owner_team,
                    "classification": classification,
                    "acl_roles": acl_roles,
                },
            )
        )
    return out

