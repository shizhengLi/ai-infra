import pytest
import torch

from minisgl.core import Req, SamplingParams


def make_req(input_len: int = 3, output_len: int = 4) -> Req:
    return Req(
        input_ids=torch.arange(input_len, dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(max_tokens=output_len),
        cache_handle=None,  # type: ignore[arg-type]
    )


def test_req_preallocates_capacity_but_exposes_valid_length() -> None:
    req = make_req(input_len=3, output_len=4)
    assert len(req.input_ids) == 3
    assert req.max_device_len == 7
    assert req.input_ids.tolist() == [0, 1, 2]
    assert req._input_buffer.numel() == 7


def test_req_append_writes_in_place_after_complete_one() -> None:
    req = make_req()
    buffer_ptr = req._input_buffer.data_ptr()

    req.complete_one()
    req.append_host(torch.tensor([10], dtype=torch.int32))
    assert req.device_len == 4
    assert len(req.input_ids) == 4
    assert req.input_ids.tolist() == [0, 1, 2, 10]
    assert req._input_buffer.data_ptr() == buffer_ptr

    req.complete_one()
    req.append_host(torch.tensor([11], dtype=torch.int32))
    assert req.device_len == 5
    assert req.input_ids.tolist() == [0, 1, 2, 10, 11]
    assert req._input_buffer.data_ptr() == buffer_ptr


def test_req_can_append_until_capacity() -> None:
    req = make_req(input_len=2, output_len=2)
    for token in (7, 8):
        req.complete_one()
        req.append_host(torch.tensor([token], dtype=torch.int32))
    assert not req.can_decode
    assert req.input_ids.tolist() == [0, 1, 7, 8]


def test_req_rejects_multiple_tokens() -> None:
    req = make_req()
    req.complete_one()
    with pytest.raises(ValueError, match="exactly one token"):
        req.append_host(torch.tensor([8, 9], dtype=torch.int32))

