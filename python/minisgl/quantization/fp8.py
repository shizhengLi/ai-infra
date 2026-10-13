from __future__ import annotations

from typing import Tuple

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_fp8_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize an [out_features, in_features] weight per output channel."""
    if weight.ndim != 2:
        raise ValueError(f"FP8 linear weight must be two-dimensional, got {weight.shape}")
    scales = (weight.float().abs().amax(dim=1) / FP8_MAX).clamp_min(1e-12)
    quantized = (weight.float() / scales[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return quantized.contiguous(), scales.contiguous()


def validate_fp8_weight(
    weight: torch.Tensor, scale: torch.Tensor, expected_shape: torch.Size
) -> Tuple[torch.Tensor, torch.Tensor]:
    if weight.dtype != FP8_DTYPE:
        raise ValueError(f"Expected {FP8_DTYPE} checkpoint weight, got {weight.dtype}")
    if weight.shape != expected_shape:
        raise ValueError(f"Expected FP8 weight shape {expected_shape}, got {weight.shape}")
    expected_scale_shape = (expected_shape[0],)
    if scale.shape != expected_scale_shape:
        raise ValueError(f"Expected FP8 scale shape {expected_scale_shape}, got {scale.shape}")
    if not scale.dtype.is_floating_point:
        raise ValueError(f"FP8 scales must be floating point, got {scale.dtype}")
    return weight.contiguous(), scale.float().contiguous()


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    from sgl_kernel import fp8_scaled_mm, sgl_per_token_quant_fp8

    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"FP8 linear requires FP16 or BF16 activations, got {x.dtype}")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(
            f"FP8 linear input dimension {x.shape[-1]} does not match weight {weight.shape[1]}"
        )

    input_shape = x.shape
    x_2d = x.reshape(-1, input_shape[-1])
    quantized_x = torch.empty_like(x_2d, dtype=FP8_DTYPE)
    input_scale = torch.empty(x_2d.shape[0], dtype=torch.float32, device=x.device)
    sgl_per_token_quant_fp8(x_2d, quantized_x, input_scale)
    output = fp8_scaled_mm(
        quantized_x,
        weight.T,
        input_scale,
        weight_scale,
        x.dtype,
        bias,
    )
    return output.reshape(*input_shape[:-1], weight.shape[0])
