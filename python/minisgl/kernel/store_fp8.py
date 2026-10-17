from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module


DEFAULT_FP8_STORE_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_fp8_module(
    element_size: int,
    input_is_bfloat16: bool,
    *,
    config: KernelConfig = DEFAULT_FP8_STORE_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, input_is_bfloat16, *config)
    return load_jit(
        "store_fp8",
        *args,
        cuda_files=["store_fp8.cu"],
        cuda_wrappers=[("launch", f"StoreFP8Kernel<{args}>::run")],
    )


def store_fp8_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Fuse BF16/FP16 -> FP8 E4M3 conversion with indexed K/V page stores."""
    if k.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"FP8 store expects float16 or bfloat16 input, got {k.dtype}")
    if v.dtype != k.dtype:
        raise TypeError(f"K/V input dtypes must match, got {k.dtype} and {v.dtype}")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise TypeError("FP8 store scale must be a one-element float32 tensor")
    num_tokens = k_cache.shape[0]
    k_cache = k_cache.view(num_tokens, -1)
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()
    module = _jit_store_fp8_module(element_size, k.dtype == torch.bfloat16)
    module.launch(k_cache, v_cache, indices, k, v, scale)
