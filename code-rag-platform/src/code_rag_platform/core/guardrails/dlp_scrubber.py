from __future__ import annotations

import re


class DLPScrubber:
    _secret_patterns = (
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA|EC|OPENSSH) PRIVATE KEY-----"),
        re.compile(r"(?i)(api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]+['\"]"),
    )
    _pii_patterns = (
        re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
        re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
        re.compile(r"\b(?:\+?\d{1,3})?[-\s]?\d{10}\b"),
    )

    def scrub(self, text: str) -> str:
        cleaned = text
        for pattern in self._secret_patterns:
            cleaned = pattern.sub("[REDACTED_SECRET]", cleaned)
        for pattern in self._pii_patterns:
            cleaned = pattern.sub("[REDACTED_PII]", cleaned)
        return cleaned

    def has_high_risk_content(self, text: str) -> bool:
        for pattern in self._secret_patterns:
            if pattern.search(text):
                return True
        return False

