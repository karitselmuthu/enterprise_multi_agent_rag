from __future__ import annotations

import argparse
import json

from .foundation.evaluation import run_prompt_regression
from .foundation.gateway import ModelGateway
from .foundation.models import InferenceRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 model gateway runner")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--task-type", type=str, default="knowledge_lookup")
    parser.add_argument("--agent-name", type=str, default="planner-agent")
    parser.add_argument("--risk-level", type=str, default="low")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--dataset", type=str, default="evals/golden_phase0.json")
    args = parser.parse_args()

    gateway = ModelGateway()

    if args.eval:
        print(json.dumps(run_prompt_regression(gateway, args.dataset), indent=2))
        return

    if not args.prompt:
        raise ValueError("--prompt is required unless --eval is set")

    result = gateway.infer(
        InferenceRequest(
            user_id="local-dev",
            agent_name=args.agent_name,
            prompt=args.prompt,
            task_type=args.task_type,
            risk_level=args.risk_level,
        )
    )
    print(json.dumps(result.__dict__, indent=2))
    print(json.dumps(gateway.telemetry.metrics.__dict__, indent=2))


if __name__ == "__main__":
    main()
