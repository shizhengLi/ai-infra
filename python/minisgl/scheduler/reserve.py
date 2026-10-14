from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass
class PageAwareReserveEstimator:
    """Estimate additional KV capacity in page-sized units.

    ``output_len`` is the request's maximum output budget.  The optional
    ratio allows a later admission controller to reserve the observed output
    fraction instead of always reserving the full maximum.
    """

    page_size: int
    new_token_ratio: float = 1.0
    min_new_token_ratio: float = 0.5
    max_new_token_ratio: float = 1.0
    update_rate: float = 0.2

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if not 0.0 < self.min_new_token_ratio <= self.max_new_token_ratio:
            raise ValueError("new-token ratio bounds must satisfy 0 < min <= max")
        if not self.min_new_token_ratio <= self.new_token_ratio <= self.max_new_token_ratio:
            raise ValueError("new_token_ratio must be between the configured bounds")
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")

    def page_count(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        return (tokens + self.page_size - 1) // self.page_size

    def estimate_additional_pages(
        self, *, cached_len: int, input_len: int, output_len: int
    ) -> int:
        if cached_len < 0 or input_len < 0 or output_len < 0:
            raise ValueError("request lengths must be non-negative")
        if cached_len > input_len:
            raise ValueError("cached_len cannot exceed input_len")
        predicted_output = ceil(output_len * self.new_token_ratio)
        # CacheManager.allocate_paged() allocates the difference between the
        # request's final page and the already cached prefix page.  Computing
        # the difference after rounding avoids over-reserving when the cached
        # prefix ends in a partially used page.
        final_pages = self.page_count(input_len + predicted_output)
        cached_pages = self.page_count(cached_len)
        return max(0, final_pages - cached_pages)

    def estimate_additional_tokens(
        self, *, cached_len: int, input_len: int, output_len: int
    ) -> int:
        return self.estimate_additional_pages(
            cached_len=cached_len, input_len=input_len, output_len=output_len
        ) * self.page_size

    def observe_output(self, *, requested_tokens: int, actual_tokens: int) -> float:
        """Update the ratio from a completed request and return the new value."""
        if requested_tokens <= 0:
            raise ValueError("requested_tokens must be positive")
        if actual_tokens < 0:
            raise ValueError("actual_tokens must be non-negative")
        observed_ratio = min(actual_tokens / requested_tokens, self.max_new_token_ratio)
        self.new_token_ratio = min(
            self.max_new_token_ratio,
            max(
                self.min_new_token_ratio,
                (1.0 - self.update_rate) * self.new_token_ratio
                + self.update_rate * observed_ratio,
            ),
        )
        return self.new_token_ratio
