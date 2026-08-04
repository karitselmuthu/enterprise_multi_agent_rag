from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .gateway import ModelGateway
from .models import InferenceRequest


@dataclass(frozen=True)
class GoldenCase:
    id: str
    prompt: str
    task_type: str
    expected_contains: tuple[str, ...]


def load_golden_dataset(path: str | Path) -> list[GoldenCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[GoldenCase] = []
    for item in raw:
        cases.append(
            GoldenCase(
                id=item["id"],
                prompt=item["prompt"],
                task_type=item["task_type"],
                expected_contains=tuple(item["expected_contains"]),
            )
        )
    return cases


def run_prompt_regression(gateway: ModelGateway, dataset_path: str | Path) -> dict[str, int]:
    passed = 0
    failed = 0
    for case in load_golden_dataset(dataset_path):
        result = gateway.infer(
            InferenceRequest(
                user_id="eval-runner",
                agent_name="verification-agent",
                prompt=case.prompt,
                task_type=case.task_type,
                risk_level="low",
            )
        )
        if all(expected in result.text for expected in case.expected_contains):
            passed += 1
        else:
            failed += 1
    return {"passed": passed, "failed": failed}

