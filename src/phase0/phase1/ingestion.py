from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..foundation.config import HardwareConfig


@dataclass(frozen=True)
class ChunkMetadata:
    source_path: str
    repo: str
    owner_team: str
    classification: str
    acl_roles: tuple[str, ...]
    content_hash: str


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata


@dataclass(frozen=True)
class ParsedNode:
    node_type: str
    text: str
    start_byte: int
    end_byte: int


class SecretScanner:
    _patterns = (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH) PRIVATE KEY-----"),
        re.compile(r"(?i)(api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]+['\"]"),
    )

    def assert_clean(self, text: str, source_path: str) -> None:
        for pattern in self._patterns:
            if pattern.search(text):
                raise ValueError(f"Secret-like content detected in {source_path}")


class ASTChunker:
    _language_by_suffix = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
    }

    def __init__(self, parser_resolver: Callable[[str], Any] | None = None) -> None:
        self._parser_resolver = parser_resolver or self._default_parser_resolver

    @staticmethod
    def _default_parser_resolver(language: str) -> Any:
        from tree_sitter_languages import get_parser  # type: ignore

        return get_parser(language)

    def _infer_language(self, path: str) -> str | None:
        suffix = Path(path).suffix.lower()
        return self._language_by_suffix.get(suffix)

    @staticmethod
    def _iter_nodes(root: Any) -> list[Any]:
        stack = [root]
        nodes: list[Any] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            children = getattr(node, "children", [])
            for child in reversed(children):
                stack.append(child)
        return nodes

    def parse_nodes(self, file_content: str, path: str) -> list[ParsedNode]:
        language = self._infer_language(path)
        if language is None:
            return []
        parser = self._parser_resolver(language)
        source_bytes = file_content.encode("utf-8")
        tree = parser.parse(source_bytes)
        selected_types = {
            "python": {"function_definition", "class_definition", "import_statement", "import_from_statement"},
            "javascript": {"function_declaration", "class_declaration", "import_statement", "method_definition"},
            "typescript": {"function_declaration", "class_declaration", "import_statement", "method_definition"},
        }[language]
        nodes: list[ParsedNode] = []
        for node in self._iter_nodes(tree.root_node):
            if node.type not in selected_types:
                continue
            text = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
            nodes.append(
                ParsedNode(
                    node_type=node.type,
                    text=text,
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                )
            )
        return sorted(nodes, key=lambda item: item.start_byte)

    def chunk(self, file_content: str, path: str, chunk_size: int = 1000) -> list[str]:
        if len(file_content) <= chunk_size:
            return [file_content]
        nodes = self.parse_nodes(file_content, path)
        if not nodes:
            return [file_content[i : i + chunk_size] for i in range(0, len(file_content), chunk_size)]

        chunks: list[str] = []
        current = ""
        for node in nodes:
            candidate = f"{current}\n{node.text}".strip() if current else node.text
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                current = node.text
                continue
            current = candidate
        if current:
            chunks.append(current)
        return chunks


class IngestionPlanner:
    def __init__(self, hardware: HardwareConfig) -> None:
        self.hardware = hardware

    def should_use_remote_vector_store(self) -> bool:
        return self.hardware.prefer_remote_qdrant or self.hardware.local_storage_budget_gb <= 256

    def incremental_targets(self, changed_paths: list[str]) -> list[str]:
        return [path for path in changed_paths if not path.endswith((".png", ".jpg", ".bin", ".lock"))]


class ChunkIndexer:
    def __init__(self, scanner: SecretScanner | None = None, chunker: ASTChunker | None = None) -> None:
        self.scanner = scanner or SecretScanner()
        self.chunker = chunker or ASTChunker()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def build_chunks(
        self,
        repo: str,
        path: str,
        owner_team: str,
        classification: str,
        acl_roles: tuple[str, ...],
        content: str,
    ) -> list[IndexedChunk]:
        self.scanner.assert_clean(content, path)
        chunks = self.chunker.chunk(content, path=path)
        indexed: list[IndexedChunk] = []
        for idx, chunk_text in enumerate(chunks):
            content_hash = self._hash_text(chunk_text)
            indexed.append(
                IndexedChunk(
                    chunk_id=f"{path}:{idx}:{content_hash[:12]}",
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        source_path=path,
                        repo=repo,
                        owner_team=owner_team,
                        classification=classification,
                        acl_roles=acl_roles,
                        content_hash=content_hash,
                    ),
                )
            )
        return indexed
