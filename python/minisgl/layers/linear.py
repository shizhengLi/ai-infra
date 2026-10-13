from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.quantization import (
    FP8_DTYPE,
    QuantizationConfig,
    awq_marlin_linear,
    fp8_linear,
    prepare_awq_marlin_weight,
    quantize_fp8_weight,
    validate_fp8_weight,
)
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        quantization_config: QuantizationConfig | None = None,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self._quantization_config = quantization_config
        if quantization_config is None:
            self.weight = torch.empty(local_osize, local_isize)
        elif quantization_config.method == "fp8":
            self.weight = torch.empty(local_osize, local_isize, dtype=FP8_DTYPE)
            self.weight_scale = torch.empty(local_osize, dtype=torch.float32)
        else:
            group_size = (
                local_isize
                if quantization_config.group_size == -1
                else quantization_config.group_size
            )
            if local_isize % group_size:
                raise ValueError(
                    f"Local input size {local_isize} is not divisible by "
                    f"AWQ group size {group_size}"
                )
            self.qweight = torch.empty(
                local_isize, local_osize // 8, dtype=torch.int32
            )
            self.scales = torch.empty(local_isize // group_size, local_osize)
            self.qzeros = torch.empty(
                local_isize // group_size, local_osize // 8, dtype=torch.int32
            )
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quantization_config is None:
            return F.linear(x, self.weight, self.bias)
        if self._quantization_config.method == "fp8":
            return fp8_linear(x, self.weight, self.weight_scale, self.bias)
        return awq_marlin_linear(
            x,
            self.qweight,
            self.scales,
            self.qzeros,
            self._marlin_workspace,
            self.local_input_size,
            self.local_output_size,
            self.bias,
        )

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if self._quantization_config is None:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)

        key_prefix = f"{prefix}." if prefix else ""
        if self._quantization_config.method == "awq":
            qweight = state_dict.pop(key_prefix + "qweight")
            scales = state_dict.pop(key_prefix + "scales")
            qzeros = state_dict.pop(key_prefix + "qzeros")
            self.qweight, self.scales, self.qzeros, self._marlin_workspace = (
                prepare_awq_marlin_weight(
                    qweight,
                    scales,
                    qzeros,
                    self.local_input_size,
                    self.local_output_size,
                    self._quantization_config.group_size,
                )
            )
            self._load_bias(state_dict, key_prefix)
            if not _internal and state_dict:
                raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
            return

        weight = state_dict.pop(key_prefix + "weight")
        checkpoint_scale = state_dict.pop(key_prefix + "weight_scale", None)
        if weight.shape != self.weight.shape:
            raise ValueError(f"Expected weight shape {self.weight.shape}, got {weight.shape}")
        if weight.dtype == FP8_DTYPE:
            if checkpoint_scale is None:
                raise ValueError(f"Missing scale for pre-quantized weight {key_prefix}weight")
            self.weight, self.weight_scale = validate_fp8_weight(
                weight, checkpoint_scale, self.weight.shape
            )
        else:
            if checkpoint_scale is not None:
                raise ValueError("Floating-point weights must not include FP8 weight_scale")
            self.weight, self.weight_scale = quantize_fp8_weight(weight)

        self._load_bias(state_dict, key_prefix)
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def _load_bias(self, state_dict: Dict[str, torch.Tensor], key_prefix: str) -> None:
        if self.bias is None:
            return
        bias = state_dict.pop(key_prefix + "bias")
        if bias.shape != self.bias.shape or bias.dtype != self.bias.dtype:
            raise ValueError(
                f"Expected bias shape/dtype {self.bias.shape}/{self.bias.dtype}, "
                f"got {bias.shape}/{bias.dtype}"
            )
        self.bias = bias


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quantization_config: QuantizationConfig | None = None,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
            quantization_config=quantization_config,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
        quantization_config: QuantizationConfig | None = None,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(
            input_size,
            output_size,
            input_size,
            tp_output_size,
            has_bias,
            quantization_config,
        )


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
        quantization_config: QuantizationConfig | None = None,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            quantization_config,
        )


class LinearOProj(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quantization_config: QuantizationConfig | None = None,
    ):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            full_isize,
            full_osize,
            local_isize,
            local_osize,
            has_bias,
            quantization_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
        quantization_config: QuantizationConfig | None = None,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(
            input_size,
            output_size,
            local_input_size,
            local_output_size,
            has_bias,
            quantization_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
