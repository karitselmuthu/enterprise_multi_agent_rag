from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .ingestion import ASTChunker


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    relation: str
    repo: str
    evidence: str


class DependencyGraphStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dependency_nodes (
                    node_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    owner_team TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dependency_edges (
                    source_node TEXT NOT NULL,
                    target_node TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    evidence TEXT NOT NULL,
                    PRIMARY KEY (source_node, target_node, relation, repo)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dependency_edges_source ON dependency_edges(source_node)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_dependency_edges_target ON dependency_edges(target_node)"
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_node(self, node_id: str, kind: str, repo: str, owner_team: str, metadata: dict[str, str]) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO dependency_nodes (node_id, kind, repo, owner_team, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    kind=excluded.kind,
                    repo=excluded.repo,
                    owner_team=excluded.owner_team,
                    metadata_json=excluded.metadata_json
                """,
                (node_id, kind, repo, owner_team, json.dumps(metadata, sort_keys=True)),
            )
            conn.commit()
        finally:
            conn.close()

    def upsert_edge(self, edge: DependencyEdge) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO dependency_edges (source_node, target_node, relation, repo, evidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source_node, target_node, relation, repo)
                DO UPDATE SET evidence=excluded.evidence
                """,
                (edge.source, edge.target, edge.relation, edge.repo, edge.evidence),
            )
            conn.commit()
        finally:
            conn.close()

    def downstream(self, node_id: str) -> list[DependencyEdge]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT source_node, target_node, relation, repo, evidence
                FROM dependency_edges
                WHERE source_node = ?
                """,
                (node_id,),
            ).fetchall()
        finally:
            conn.close()
        return [DependencyEdge(*row) for row in rows]


class DependencyGraphBuilder:
    _py_import_re = re.compile(r"^\s*import\s+([a-zA-Z0-9_\.]+)", re.MULTILINE)
    _py_from_re = re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+", re.MULTILINE)
    _js_import_re = re.compile(r"^\s*import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE)

    def __init__(self, parser: ASTChunker | None = None) -> None:
        self.parser = parser or ASTChunker()

    def extract_edges(self, repo: str, module_id: str, path: str, content: str) -> list[DependencyEdge]:
        try:
            nodes = self.parser.parse_nodes(content, path)
        except ModuleNotFoundError as exc:
            if exc.name not in {"tree_sitter_languages", "tree_sitter"}:
                raise
            nodes = []
        imports = [
            node.text
            for node in nodes
            if node.node_type in {"import_statement", "import_from_statement"}
        ]
        joined = "\n".join(imports) if imports else content
        dependencies = self._extract_import_targets(joined, path)
        return [
            DependencyEdge(
                source=module_id,
                target=target,
                relation="imports",
                repo=repo,
                evidence=f"{Path(path).name}: {target}",
            )
            for target in dependencies
        ]

    def _extract_import_targets(self, text: str, path: str) -> list[str]:
        targets: list[str] = []
        if path.endswith(".py"):
            targets.extend(self._py_import_re.findall(text))
            targets.extend(self._py_from_re.findall(text))
        else:
            targets.extend(self._js_import_re.findall(text))
        unique: list[str] = []
        seen = set()
        for target in targets:
            if target in seen:
                continue
            seen.add(target)
            unique.append(target)
        return unique
