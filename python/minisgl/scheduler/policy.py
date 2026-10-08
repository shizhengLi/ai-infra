from __future__ import annotations

from dataclasses import dataclass


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

    def __post_init__(self) -> None:
        if self.default_budget <= 0:
            raise ValueError("default prefill budget must be positive")
        if not 0 <= self.decode_active_budget <= self.default_budget:
            raise ValueError("decode-active prefill budget must be between zero and default")

    def select(self, *, decode_runnable: bool) -> int:
        if decode_runnable and self.decode_active_budget > 0:
            return self.decode_active_budget
        return self.default_budget
