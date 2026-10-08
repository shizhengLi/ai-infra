from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "benchmark" / "online" / "bench_poisson_l20.py"
SPEC = spec_from_file_location("bench_poisson_l20", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
generate_arrival_offsets = MODULE.generate_arrival_offsets
generate_balanced_input_lengths = MODULE.generate_balanced_input_lengths
summarize_run = MODULE.summarize_run
RawResult = MODULE.RawResult


def test_poisson_arrivals_are_deterministic_and_increasing() -> None:
    first = generate_arrival_offsets(rate=1.0, count=16, seed=42)
    second = generate_arrival_offsets(rate=1.0, count=16, seed=42)

    assert first == second
    assert first[0] == 0.0
    assert len(first) == 16
    assert all(left < right for left, right in zip(first, first[1:]))


def test_poisson_arrivals_reject_invalid_parameters() -> None:
    for rate, count in ((0.0, 1), (-1.0, 1), (1.0, 0)):
        try:
            generate_arrival_offsets(rate=rate, count=count, seed=42)
        except ValueError as exc:
            assert "must be positive" in str(exc)
        else:
            raise AssertionError(f"invalid parameters were accepted: rate={rate}, count={count}")


def test_balanced_input_lengths_are_deterministic_and_balanced() -> None:
    first = generate_balanced_input_lengths([256, 1024, 4096], count=24, seed=43)
    second = generate_balanced_input_lengths([256, 1024, 4096], count=24, seed=43)

    assert first == second
    assert first != sorted(first)
    assert {length: first.count(length) for length in set(first)} == {
        256: 8,
        1024: 8,
        4096: 8,
    }


def test_balanced_input_lengths_reject_invalid_parameters() -> None:
    invalid = (([], 1), ([0, 1024], 2), ([256, 256], 2), ([256], 0))
    for input_lengths, count in invalid:
        try:
            generate_balanced_input_lengths(input_lengths, count=count, seed=43)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid parameters were accepted: lengths={input_lengths}, count={count}"
            )


def test_run_summary_groups_latency_by_input_length() -> None:
    raw = [
        RawResult(input_len=256, output_len=2, message="a", tics=[0.0, 0.1, 0.2]),
        RawResult(input_len=4096, output_len=2, message="b", tics=[1.0, 1.4, 1.6]),
    ]

    summary = summarize_run(
        raw,
        offsets=[0.0, 1.0],
        input_lengths=[256, 4096],
        output_len=2,
        ttft_slo_ms=2000.0,
        tpot_slo_ms=1000.0,
    )

    assert summary["ttft_avg_by_input_len_ms"] == pytest.approx(
        {"256": 100.0, "4096": 400.0}
    )
    assert summary["ttft_p90_by_input_len_ms"] == pytest.approx(
        {"256": 100.0, "4096": 400.0}
    )
    assert summary["e2e_p90_by_input_len_s"] == pytest.approx(
        {"256": 0.2, "4096": 0.6}
    )
