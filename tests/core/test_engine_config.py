import torch

from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig


def test_scheduler_forward_bound_uses_prefill_budget() -> None:
    config = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(rank=0, size=4),
        dtype=torch.bfloat16,
        max_extend_tokens=8192,
        max_seq_len_override=40960,
    )

    assert config.max_forward_len == 8192
    assert config.max_forward_len != config.max_seq_len
