from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefillSchedulePolicy:
    max_prefill_streak: int = 0
    prefill_streak: int = 0

    def __post_init__(self) -> None:
        if self.max_prefill_streak < 0:
            raise ValueError("max_prefill_streak must be non-negative")

    def prefer_decode(self, *, prefill_runnable: bool, decode_runnable: bool) -> bool:
        return (
            self.max_prefill_streak > 0
            and self.prefill_streak >= self.max_prefill_streak
            and prefill_runnable
            and decode_runnable
        )

    def update(self, *, is_prefill: bool) -> None:
        if is_prefill and self.max_prefill_streak > 0:
            self.prefill_streak = min(self.prefill_streak + 1, self.max_prefill_streak)
        else:
            self.prefill_streak = 0


@dataclass(frozen=True)
class PrefillBudgetPolicy:
    default_budget: int
    decode_active_budget: int = 0
    decode_overload_budget: int = 0
    decode_overload_threshold: int = 0

    def __post_init__(self) -> None:
        if self.default_budget <= 0:
            raise ValueError("default prefill budget must be positive")
        if not 0 <= self.decode_active_budget <= self.default_budget:
            raise ValueError("decode-active prefill budget must be between zero and default")
        active_budget = self.decode_active_budget or self.default_budget
        if not 0 <= self.decode_overload_budget <= active_budget:
            raise ValueError("decode-overload prefill budget must be between zero and active")
        overload_enabled = self.decode_overload_budget > 0
        if overload_enabled != (self.decode_overload_threshold > 0):
            raise ValueError("decode-overload budget and threshold must be enabled together")

    def select(self, *, decode_runnable: bool, pending_prefill_tokens: int = 0) -> int:
        if pending_prefill_tokens < 0:
            raise ValueError("pending prefill tokens must be non-negative")
        if (
            decode_runnable
            and self.decode_overload_budget > 0
            and pending_prefill_tokens >= self.decode_overload_threshold
        ):
            return self.decode_overload_budget
        if decode_runnable and self.decode_active_budget > 0:
            return self.decode_active_budget
        return self.default_budget


@dataclass
class PrefillTelemetry:
    budget_selections: dict[int, int] = field(default_factory=dict)
    prefill_batches: int = 0
    admitted_prefill_tokens: int = 0
    budget_limited_batches: int = 0
    max_pending_prefill_tokens: int = 0
    max_pending_prefill_requests: int = 0

    def record_selection(self, budget: int, pending_tokens: int, pending_requests: int) -> None:
        self.budget_selections[budget] = self.budget_selections.get(budget, 0) + 1
        self.max_pending_prefill_tokens = max(self.max_pending_prefill_tokens, pending_tokens)
        self.max_pending_prefill_requests = max(self.max_pending_prefill_requests, pending_requests)

    def record_batch(self, admitted_tokens: int, budget_limited: bool) -> None:
        self.prefill_batches += 1
        self.admitted_prefill_tokens += admitted_tokens
        self.budget_limited_batches += int(budget_limited)

    def snapshot(self) -> dict[str, int | dict[str, int]]:
        return {
            "budget_selections": {
                str(budget): count for budget, count in sorted(self.budget_selections.items())
            },
            "prefill_batches": self.prefill_batches,
            "admitted_prefill_tokens": self.admitted_prefill_tokens,
            "budget_limited_batches": self.budget_limited_batches,
            "max_pending_prefill_tokens": self.max_pending_prefill_tokens,
            "max_pending_prefill_requests": self.max_pending_prefill_requests,
        }
