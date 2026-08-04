import unittest
from pathlib import Path

from phase0.foundation.evaluation import run_prompt_regression
from phase0.foundation.gateway import ModelGateway


class EvaluationTests(unittest.TestCase):
    def test_prompt_regression(self) -> None:
        gateway = ModelGateway()
        dataset = Path(__file__).resolve().parents[1] / "evals" / "golden_phase0.json"
        summary = run_prompt_regression(gateway, dataset)
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["passed"], 2)


if __name__ == "__main__":
    unittest.main()
