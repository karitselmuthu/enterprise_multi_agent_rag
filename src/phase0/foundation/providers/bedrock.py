from __future__ import annotations

import time

from .base import BaseProvider
from ..models import InferenceRequest, InferenceResult


class BedrockProvider(BaseProvider):
    name = "bedrock"

    def complete(self, request: InferenceRequest, model: str | None = None) -> InferenceResult:
        selected_model = model or self.default_model
        start = time.perf_counter()
        response = f"[bedrock:{selected_model}] {request.prompt[:240]}"
        return self._build_result(response, request, selected_model, start)

