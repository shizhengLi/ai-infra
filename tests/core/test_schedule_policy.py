from minisgl.scheduler.policy import PrefillBudgetPolicy, PrefillSchedulePolicy


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
