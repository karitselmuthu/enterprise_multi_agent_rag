from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    precision: float
    recall: float
    mrr: float
    citation_quality: float


@dataclass(frozen=True)
class AnswerMetrics:
    groundedness: float
    completeness: float
    latency_ms: float
    cost_usd: float


def compute_retrieval_metrics(
    relevant_retrieved: int,
    total_retrieved: int,
    total_relevant: int,
    reciprocal_rank: float,
    citation_quality: float,
) -> RetrievalMetrics:
    precision = relevant_retrieved / total_retrieved if total_retrieved else 0.0
    recall = relevant_retrieved / total_relevant if total_relevant else 0.0
    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        mrr=reciprocal_rank,
        citation_quality=citation_quality,
    )


def compute_answer_metrics(groundedness: float, completeness: float, latency_ms: float, cost_usd: float) -> AnswerMetrics:
    return AnswerMetrics(
        groundedness=groundedness,
        completeness=completeness,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


class CostAwareRouterTuner:
    def __init__(self, max_cost_per_request_usd: float, max_latency_ms: float) -> None:
        self.max_cost_per_request_usd = max_cost_per_request_usd
        self.max_latency_ms = max_latency_ms

    def should_downgrade_model(self, avg_cost_usd: float, avg_latency_ms: float, quality_score: float) -> bool:
        if quality_score < 0.8:
            return False
        return avg_cost_usd > self.max_cost_per_request_usd or avg_latency_ms > self.max_latency_ms
