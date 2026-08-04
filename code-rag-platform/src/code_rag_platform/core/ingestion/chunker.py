from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaggedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | list[str]]


class TreeSitterChunker:
    _language_by_suffix = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }

    @staticmethod
    def _get_parser(language: str) -> Any:
        try:
            from tree_sitter_languages import get_parser  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "tree_sitter_languages is required for AST chunking. Install requirements.txt"
            ) from exc
        return get_parser(language)

    @classmethod
    def _language_for_path(cls, path: str) -> str | None:
        suffix = Path(path).suffix.lower()
        return cls._language_by_suffix.get(suffix)

    @staticmethod
    def _walk(root: Any) -> list[Any]:
        stack = [root]
        out: list[Any] = []
        while stack:
            node = stack.pop()
            out.append(node)
            for child in reversed(getattr(node, "children", [])):
                stack.append(child)
        return out

    def chunk(self, path: str, content: str, chunk_size: int = 1200) -> list[str]:
        language = self._language_for_path(path)
        segments: list[str] = []
        if language is not None:
            parser = self._get_parser(language)
            raw = content.encode("utf-8")
            tree = parser.parse(raw)
            selected_types = {
                "python": {"function_definition", "class_definition", "import_statement", "import_from_statement"},
                "javascript": {"function_declaration", "class_declaration", "import_statement", "method_definition"},
                "typescript": {"function_declaration", "class_declaration", "import_statement", "method_definition"},
            }[language]
            for node in self._walk(tree.root_node):
                if node.type in selected_types:
                    text = raw[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
                    if text.strip():
                        segments.append(text)
        if not segments:
            segments = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]

        chunks: list[str] = []
        current = ""
        for segment in segments:
            candidate = f"{current}\n{segment}".strip() if current else segment
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                current = segment
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks


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

