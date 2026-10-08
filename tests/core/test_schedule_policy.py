from minisgl.scheduler.policy import PrefillBudgetPolicy, PrefillSchedulePolicy, PrefillTelemetry


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

    assert telemetry.snapshot() == {
        "budget_selections": {"2048": 1, "4096": 1},
        "prefill_batches": 2,
        "admitted_prefill_tokens": 5048,
        "budget_limited_batches": 1,
        "max_pending_prefill_tokens": 5000,
        "max_pending_prefill_requests": 5,
    }
