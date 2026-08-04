from __future__ import annotations

import time

from .base import BaseProvider
from ..models import InferenceRequest, InferenceResult


class LocalProvider(BaseProvider):
    name = "local"

    def __init__(self, default_model: str) -> None:
        super().__init__(default_model)
        self._loaded_model: str | None = None
        self.model_swaps = 0

    def complete(self, request: InferenceRequest, model: str | None = None) -> InferenceResult:
        selected_model = model or self.default_model
        start = time.perf_counter()
        if self._loaded_model and self._loaded_model != selected_model:
            self.model_swaps += 1
        self._loaded_model = selected_model
        response = f"[local:{selected_model}] {request.prompt[:240]}"
        return self._build_result(response, request, selected_model, start)
