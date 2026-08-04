from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from .config import BudgetConfig, RateLimitConfig


class RateLimitExceededError(RuntimeError):
    pass


class BudgetExceededError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, config: RateLimitConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def assert_allowed(self, key: str) -> None:
        now = time.time()
        window_start = now - 60
        with self._lock:
            events = self._events[key]
            while events and events[0] < window_start:
                events.popleft()
            if len(events) >= self._config.requests_per_minute + self._config.burst:
                raise RateLimitExceededError(f"Rate limit exceeded for key={key}")
            events.append(now)


class BudgetTracker:
    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._user_cost: dict[tuple[str, str], float] = defaultdict(float)
        self._team_cost: dict[tuple[str, str], float] = defaultdict(float)

    @staticmethod
    def _date_key(epoch_seconds: float) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(epoch_seconds))

    def assert_within_budget(self, user_id: str, team_id: str, expected_cost: float) -> None:
        date_key = self._date_key(time.time())
        with self._lock:
            user_spend = self._user_cost[(date_key, user_id)]
            team_spend = self._team_cost[(date_key, team_id)]
            if user_spend + expected_cost > self._config.user_daily_usd:
                raise BudgetExceededError(f"User daily budget exceeded for user={user_id}")
            if team_spend + expected_cost > self._config.team_daily_usd:
                raise BudgetExceededError(f"Team daily budget exceeded for team={team_id}")

    def record_spend(self, user_id: str, team_id: str, cost_usd: float) -> None:
        date_key = self._date_key(time.time())
        with self._lock:
            self._user_cost[(date_key, user_id)] += cost_usd
            self._team_cost[(date_key, team_id)] += cost_usd
