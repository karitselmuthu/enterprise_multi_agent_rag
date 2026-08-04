from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AccessContext:
    user_id: str
    roles: tuple[str, ...]
    attributes: dict[str, str]
    repo_entitlements: tuple[str, ...]
    clearance: str


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    repo: str
    classification: str
    acl_roles: tuple[str, ...]
    text: str


class RetrievalGuard:
    _clearance_order = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

    def filter_chunks(self, context: AccessContext, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        allowed: list[EvidenceChunk] = []
        max_level = self._clearance_order.get(context.clearance, -1)
        for chunk in chunks:
            if chunk.repo not in context.repo_entitlements:
                continue
            chunk_level = self._clearance_order.get(chunk.classification, 99)
            if chunk_level > max_level:
                continue
            if not set(chunk.acl_roles).intersection(context.roles):
                continue
            allowed.append(chunk)
        return allowed


class ToolGuard:
    def __init__(self, allowed_tools: dict[str, tuple[str, ...]]) -> None:
        self.allowed_tools = allowed_tools

    def validate(self, tool_name: str, params: dict[str, Any], roles: tuple[str, ...]) -> None:
        permitted_roles = self.allowed_tools.get(tool_name)
        if not permitted_roles:
            raise PermissionError(f"Tool {tool_name} is not allowlisted")
        if not set(roles).intersection(permitted_roles):
            raise PermissionError(f"User lacks permission for tool {tool_name}")
        if "command" in params and ("rm -rf" in str(params["command"]) or "DROP TABLE" in str(params["command"]).upper()):
            raise PermissionError("Unsafe tool parameters rejected")


class ImmutableAuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_entry(entry: dict[str, Any], prev_hash: str) -> str:
        payload = json.dumps({"prev_hash": prev_hash, "entry": entry}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def append(self, entry: dict[str, Any]) -> None:
        prev_hash = "GENESIS"
        if self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            if lines:
                prev = json.loads(lines[-1])
                prev_hash = prev["hash"]
        current_hash = self._hash_entry(entry, prev_hash)
        record = {"prev_hash": prev_hash, "hash": current_hash, "entry": entry}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
