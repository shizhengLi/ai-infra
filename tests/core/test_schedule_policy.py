from types import SimpleNamespace

from minisgl.scheduler.policy import PrefillBudgetPolicy, PrefillSchedulePolicy, PrefillTelemetry
from minisgl.scheduler.scheduler import Scheduler


def _guard_scheduler(*, guard_size: int, padded_size: int, prefill_runnable: bool):
    batch = SimpleNamespace(
        is_decode=True,
        padded_size=padded_size,
        padded_reqs=(),
    )
    graph_runner = SimpleNamespace(can_use_cuda_graph=lambda candidate: candidate is batch)
    return SimpleNamespace(
        decode_result_before_prefill=False,
        decode_graph_tail_prefill_batch_size=guard_size,
        _decode_graph_tail_priority_consumed=False,
        prefill_manager=SimpleNamespace(runnable=prefill_runnable),
        decode_manager=SimpleNamespace(runnable=True),
        engine=SimpleNamespace(graph_runner=graph_runner),
    ), (SimpleNamespace(batch=batch), None)


def test_graph_tail_guard_matches_padded_batch() -> None:
    scheduler, last_data = _guard_scheduler(
        guard_size=32, padded_size=32, prefill_runnable=True
    )
    assert Scheduler._should_process_decode_before_prefill(scheduler, last_data)


def test_graph_tail_guard_ignores_other_batch_sizes() -> None:
    scheduler, last_data = _guard_scheduler(
        guard_size=32, padded_size=24, prefill_runnable=True
    )
    assert not Scheduler._should_process_decode_before_prefill(scheduler, last_data)


def test_graph_tail_guard_requires_pending_prefill() -> None:
    scheduler, last_data = _guard_scheduler(
        guard_size=32, padded_size=32, prefill_runnable=False
    )
    assert not Scheduler._should_process_decode_before_prefill(scheduler, last_data)


def test_graph_tail_guard_zero_preserves_default_overlap() -> None:
    scheduler, last_data = _guard_scheduler(
        guard_size=0, padded_size=32, prefill_runnable=True
    )
    assert not Scheduler._should_process_decode_before_prefill(scheduler, last_data)


def test_graph_tail_priority_matches_padded_batch_without_synchronizing() -> None:
    scheduler, last_data = _guard_scheduler(
        guard_size=0, padded_size=32, prefill_runnable=True
    )
    scheduler.decode_graph_tail_prefill_priority_batch_size = 32
    assert Scheduler._should_prioritize_decode_before_prefill(scheduler, last_data)
    scheduler._decode_graph_tail_priority_consumed = True
    assert not Scheduler._should_prioritize_decode_before_prefill(scheduler, last_data)


def test_graph_tail_priority_requires_decode_runnable() -> None:
    scheduler, last_data = _guard_scheduler(
        guard_size=0, padded_size=32, prefill_runnable=True
    )
    scheduler.decode_graph_tail_prefill_priority_batch_size = 32
    scheduler.decode_manager.runnable = False
    assert not Scheduler._should_prioritize_decode_before_prefill(scheduler, last_data)


def test_unlimited_prefill_priority() -> None:
    policy = PrefillSchedulePolicy(max_prefill_streak=0)
    for _ in range(3):
        policy.update(is_prefill=True)
    assert policy.prefill_streak == 0
    assert not policy.prefer_decode(prefill_runnable=True, decode_runnable=True)


def test_bounded_prefill_streak_interleaves_decode() -> None:
    policy = PrefillSchedulePolicy(max_prefill_streak=1)
    assert not policy.prefer_decode(prefill_runnable=True, decode_runnable=True)

    policy.update(is_prefill=True)
    assert policy.prefer_decode(prefill_runnable=True, decode_runnable=True)
    policy.update(is_prefill=True)
    assert policy.prefill_streak == 1

    policy.update(is_prefill=False)
    assert policy.prefill_streak == 0
    assert not policy.prefer_decode(prefill_runnable=True, decode_runnable=True)


def test_prefill_limit_does_not_block_when_decode_is_idle() -> None:
    policy = PrefillSchedulePolicy(max_prefill_streak=1)
    policy.update(is_prefill=True)
    assert not policy.prefer_decode(prefill_runnable=True, decode_runnable=False)
    assert not policy.prefer_decode(prefill_runnable=False, decode_runnable=True)


def test_negative_prefill_streak_is_rejected() -> None:
    try:
        PrefillSchedulePolicy(max_prefill_streak=-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative max_prefill_streak was accepted")


def test_decode_active_prefill_budget() -> None:
    policy = PrefillBudgetPolicy(default_budget=8192, decode_active_budget=4096)
    assert policy.select(decode_runnable=False) == 8192
    assert policy.select(decode_runnable=True) == 4096


def test_zero_decode_active_budget_reuses_default() -> None:
    policy = PrefillBudgetPolicy(default_budget=8192)
    assert policy.select(decode_runnable=False) == 8192
    assert policy.select(decode_runnable=True) == 8192


def test_default_prefill_budget_must_be_positive() -> None:
    try:
        PrefillBudgetPolicy(default_budget=0)
    except ValueError as exc:
        assert "must be positive" in str(exc)
    else:
        raise AssertionError("non-positive default prefill budget was accepted")


def test_decode_active_budget_cannot_exceed_default() -> None:
    try:
        PrefillBudgetPolicy(default_budget=8192, decode_active_budget=16384)
    except ValueError as exc:
        assert "between zero and default" in str(exc)
    else:
        raise AssertionError("oversized decode-active prefill budget was accepted")


def test_decode_overload_prefill_budget_uses_queue_pressure() -> None:
    policy = PrefillBudgetPolicy(
        default_budget=8192,
        decode_active_budget=4096,
        decode_overload_budget=2048,
        decode_overload_threshold=4096,
    )
    assert policy.select(decode_runnable=False, pending_prefill_tokens=8192) == 8192
    assert policy.select(decode_runnable=True, pending_prefill_tokens=4095) == 4096
    assert policy.select(decode_runnable=True, pending_prefill_tokens=4096) == 2048


def test_decode_overload_budget_and_threshold_are_enabled_together() -> None:
    for budget, threshold in ((2048, 0), (0, 4096)):
        try:
            PrefillBudgetPolicy(
                default_budget=8192,
                decode_active_budget=4096,
                decode_overload_budget=budget,
                decode_overload_threshold=threshold,
            )
        except ValueError as exc:
            assert "enabled together" in str(exc)
        else:
            raise AssertionError("partial overload configuration was accepted")


def test_decode_overload_budget_cannot_exceed_active_budget() -> None:
    try:
        PrefillBudgetPolicy(
            default_budget=8192,
            decode_active_budget=4096,
            decode_overload_budget=8192,
            decode_overload_threshold=4096,
        )
    except ValueError as exc:
        assert "between zero and active" in str(exc)
    else:
        raise AssertionError("oversized decode-overload prefill budget was accepted")


def test_pending_prefill_tokens_cannot_be_negative() -> None:
    policy = PrefillBudgetPolicy(default_budget=8192)
    try:
        policy.select(decode_runnable=True, pending_prefill_tokens=-1)
    except ValueError as exc:
        assert "must be non-negative" in str(exc)
    else:
        raise AssertionError("negative pending prefill tokens were accepted")


def test_prefill_telemetry_tracks_budget_binding() -> None:
    telemetry = PrefillTelemetry()
    telemetry.record_selection(4096, pending_tokens=3000, pending_requests=3)
    telemetry.record_batch(admitted_tokens=3000, budget_limited=False)
    telemetry.record_selection(2048, pending_tokens=5000, pending_requests=5)
    telemetry.record_batch(admitted_tokens=2048, budget_limited=True)
    telemetry.record_execution(
        selected_budget=2048,
        admitted_tokens=2048,
        active_decode_requests=4,
        execution_ms=612.5,
    )

    assert telemetry.snapshot() == {
        "budget_selections": {"2048": 1, "4096": 1},
        "prefill_batches": 2,
        "admitted_prefill_tokens": 5048,
        "budget_limited_batches": 1,
        "max_pending_prefill_tokens": 5000,
        "max_pending_prefill_requests": 5,
        "batch_execution_samples": [
            {
                "selected_budget": 2048,
                "admitted_tokens": 2048,
                "active_decode_requests": 4,
                "execution_ms": 612.5,
            }
        ],
    }

    telemetry.drain_execution_samples()
    assert telemetry.snapshot()["batch_execution_samples"] == []
