from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.quantization import QuantizationConfig
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    attention_backend: str = "auto"
    moe_backend: str = "auto"
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    cuda_graph_disable_bs: List[int] | None = None
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    allow_bf16_reduced_precision_reduction: bool = True
    quantization: str | None = None
    kv_cache_dtype: str = "auto"
    kv_cache_scale: float = 1.0
    kv_cache_calibration: str = "fixed"
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig

        config = ModelConfig.from_hf(self.hf_config)
        return replace(config, quantization_config=self.quantization_config)

    @cached_property
    def quantization_config(self) -> QuantizationConfig | None:
        checkpoint_config = (
            getattr(self.hf_config, "quantization_config", None)
            if self.quantization == "awq"
            else None
        )
        return QuantizationConfig.from_name(self.quantization, checkpoint_config)

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
