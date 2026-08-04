from __future__ import annotations

from dataclasses import dataclass


RISKY_ACTION_CATEGORIES = {
    "production_change",
    "database_change",
    "infrastructure_change",
    "security_change",
}


@dataclass(frozen=True)
class ActionRequest:
    category: str
    description: str
    requested_by: str


class HumanApprovalGate:
    def requires_approval(self, action: ActionRequest) -> bool:
        return action.category in RISKY_ACTION_CATEGORIES


class RolloutManager:
    _allowed_stages = ("pilot", "department", "enterprise")

    def __init__(self) -> None:
        self._team_stage: dict[str, str] = {}

    def set_stage(self, team: str, stage: str) -> None:
        if stage not in self._allowed_stages:
            raise ValueError(f"Invalid rollout stage: {stage}")
        self._team_stage[team] = stage

    def get_stage(self, team: str) -> str:
        return self._team_stage.get(team, "pilot")
