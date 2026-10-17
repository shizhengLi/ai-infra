import torch

from minisgl.core import Req, SamplingParams
from minisgl.scheduler.decode import DecodeManager


def make_req(uid: int, *, input_len: int = 8, output_len: int = 64) -> Req:
    return Req(
        input_ids=torch.arange(input_len, dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=output_len,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,  # type: ignore[arg-type]
    )


def test_retraction_prefers_largest_page_demand_and_respects_inflight() -> None:
    manager = DecodeManager(page_size=16)
    small = make_req(1, output_len=17)
    large = make_req(2, output_len=64)
    manager.running_reqs = {small, large}

    assert manager.retract_candidate({large}) is small
    assert manager.retract_candidate() is large
    assert not manager.runnable


def test_retraction_pause_resume_preserves_generated_history() -> None:
    req = make_req(7, input_len=3, output_len=4)
    req.complete_one()
    req.append_host(torch.tensor([11], dtype=torch.int32))
    req.complete_one()
    req.append_host(torch.tensor([12], dtype=torch.int32))
    manager = DecodeManager(page_size=16, running_reqs={req})
    assert manager.pause_candidate() is req
    assert req.retracted
    assert manager.paused_reqs == [req]
    manager.resume_paused()
    assert manager.running_reqs == {req}
    assert req.input_ids.tolist() == [0, 1, 2, 11, 12]
    assert req.remain_len == 2
    assert req.generation == 2
    assert not req.retracted


def test_paused_request_keeps_one_page_reservation() -> None:
    req = make_req(3, output_len=64)
    manager = DecodeManager(page_size=16, running_reqs={req})
    full_reservation = manager.inflight_tokens
    assert manager.pause_candidate() is req
    assert manager.inflight_tokens == 16
    assert manager.inflight_tokens < full_reservation


def test_resumed_request_obeys_retraction_cooldown() -> None:
    req = make_req(4)
    manager = DecodeManager(page_size=16, retraction_cooldown_steps=2, running_reqs={req})
    assert manager.pause_candidate() is req
    manager.resume_paused()
    assert manager.retract_candidate() is None
    manager.advance_retraction_cooldown()
    assert manager.retract_candidate() is None
    manager.advance_retraction_cooldown()
    assert manager.retract_candidate() is req


def test_abort_clears_retraction_cooldown() -> None:
    req = make_req(4)
    manager = DecodeManager(page_size=16, retraction_cooldown_steps=2, running_reqs={req})
    manager._retraction_cooldowns[req] = 2

    assert manager.abort_req(req.uid) is req
    assert req not in manager._retraction_cooldowns
