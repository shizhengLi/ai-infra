import pytest

from minisgl.scheduler.reserve import PageAwareReserveEstimator


def test_page_aware_reservation_rounds_to_pages() -> None:
    estimator = PageAwareReserveEstimator(page_size=16)
    assert estimator.estimate_additional_pages(
        cached_len=32, input_len=65, output_len=17
    ) == 4
    assert estimator.estimate_additional_tokens(
        cached_len=32, input_len=65, output_len=17
    ) == 64


def test_page_aware_reservation_accounts_for_prefix_cache() -> None:
    estimator = PageAwareReserveEstimator(page_size=16)
    assert estimator.estimate_additional_tokens(
        cached_len=64, input_len=100, output_len=16
    ) == 64
    assert estimator.estimate_additional_tokens(
        cached_len=0, input_len=100, output_len=16
    ) == 128


def test_partial_cached_page_is_not_over_reserved() -> None:
    estimator = PageAwareReserveEstimator(page_size=4)
    # Both the cached prefix [0, 5) and the final sequence [0, 7) occupy
    # two pages, so no additional page is needed for the one-token extension.
    assert estimator.estimate_additional_pages(
        cached_len=5, input_len=6, output_len=1
    ) == 0


def test_page_size_one_matches_legacy_estimate() -> None:
    estimator = PageAwareReserveEstimator(page_size=1)
    assert estimator.estimate_additional_tokens(
        cached_len=13, input_len=37, output_len=29
    ) == 53


def test_ratio_reduces_reservation_and_updates_with_ema() -> None:
    estimator = PageAwareReserveEstimator(
        page_size=16, new_token_ratio=0.5, min_new_token_ratio=0.1
    )
    assert estimator.estimate_additional_tokens(
        cached_len=0, input_len=16, output_len=32
    ) == 32
    assert estimator.observe_output(requested_tokens=32, actual_tokens=0) == pytest.approx(0.4)
    assert estimator.new_token_ratio >= estimator.min_new_token_ratio


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page_size": 0},
        {"page_size": 16, "new_token_ratio": 0.4, "min_new_token_ratio": 0.5},
        {"page_size": 16, "update_rate": 0.0},
    ],
)
def test_invalid_estimator_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        PageAwareReserveEstimator(**kwargs)


def test_cli_selects_page_aware_policy() -> None:
    from minisgl.server.args import parse_args

    args, _ = parse_args(
        [
            "--model",
            "unused",
            "--dtype",
            "bfloat16",
            "--page-size",
            "16",
            "--prefill-reserve-policy",
            "page-aware",
            "--prefill-new-token-ratio",
            "0.75",
        ]
    )
    assert args.prefill_reserve_policy == "page-aware"
    assert args.prefill_new_token_ratio == pytest.approx(0.75)
