from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "benchmark" / "online" / "bench_poisson_l20.py"
SPEC = spec_from_file_location("bench_poisson_l20", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
generate_arrival_offsets = MODULE.generate_arrival_offsets


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
