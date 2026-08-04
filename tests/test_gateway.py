import unittest

from phase0.foundation.config import BudgetConfig, GatewayConfig, HardwareConfig, ProviderConfig
from phase0.foundation.gateway import ModelGateway
from phase0.foundation.models import InferenceRequest
from phase0.foundation.policies import BudgetExceededError, RateLimitExceededError


class GatewayTests(unittest.TestCase):
    def test_local_route_for_low_risk_task(self) -> None:
        gateway = ModelGateway()
        result = gateway.infer(
            InferenceRequest(
                user_id="u1",
                agent_name="retriever",
                prompt="find ownership for payments service",
                task_type="ownership_lookup",
            )
        )
        self.assertEqual(result.provider, "local")

    def test_cloud_route_for_security_task(self) -> None:
        gateway = ModelGateway()
        result = gateway.infer(
            InferenceRequest(
                user_id="u2",
                agent_name="verification",
                prompt="security review authz middleware",
                task_type="security_review",
                risk_level="high",
            )
        )
        self.assertIn(result.provider, {"bedrock", "vertex"})

    def test_rate_limit_enforced(self) -> None:
        config = GatewayConfig()
        gateway = ModelGateway(config=config)
        request = InferenceRequest(
            user_id="u3",
            agent_name="planner",
            prompt="p",
            task_type="ownership_lookup",
        )
        # Initial request is okay and populates prompt cache; subsequent cache hits still count for rate limiting.
        gateway.infer(request)
        for _ in range(config.rate_limit.requests_per_minute + config.rate_limit.burst - 1):
            gateway.infer(request)
        with self.assertRaises(RateLimitExceededError):
            gateway.infer(request)

    def test_budget_enforced(self) -> None:
        config = GatewayConfig(
            local=ProviderConfig(True, "qwen-14b"),
            bedrock=ProviderConfig(True, "claude"),
            vertex=ProviderConfig(True, "gemini"),
            budget=BudgetConfig(user_daily_usd=0.00001, team_daily_usd=0.00001),
        )
        gateway = ModelGateway(config=config)
        with self.assertRaises(BudgetExceededError):
            gateway.infer(
                InferenceRequest(
                    user_id="u4",
                    agent_name="planner",
                    prompt="This call should exceed budget immediately",
                    task_type="knowledge_lookup",
                )
            )

    def test_8gb_profile_prefers_small_local_model(self) -> None:
        config = GatewayConfig(hardware=HardwareConfig(system_ram_gb=8, available_model_ram_gb=4.0))
        gateway = ModelGateway(config=config)
        result = gateway.infer(
            InferenceRequest(
                user_id="u5",
                agent_name="summarizer",
                prompt="summarize service ownership changes",
                task_type="ownership_lookup",
            )
        )
        self.assertEqual(result.provider, "local")
        self.assertEqual(result.model, "qwen2.5-coder-1.5b-instruct-q4")

    def test_8gb_profile_forces_cloud_for_deep_tasks(self) -> None:
        config = GatewayConfig(hardware=HardwareConfig(system_ram_gb=8, available_model_ram_gb=4.0))
        gateway = ModelGateway(config=config)
        result = gateway.infer(
            InferenceRequest(
                user_id="u6",
                agent_name="incident-agent",
                prompt="perform incident RCA for distributed payment timeout",
                task_type="incident_rca",
                risk_level="high",
            )
        )
        self.assertIn(result.provider, {"bedrock", "vertex"})


if __name__ == "__main__":
    unittest.main()
