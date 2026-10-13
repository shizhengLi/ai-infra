from __future__ import annotations

from typing import Tuple

import torch


INT4_BITS = 4
INT4_PACK_FACTOR = 8


def _unpack_cols(packed: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    expected = (size_k, size_n // INT4_PACK_FACTOR)
    if packed.dtype != torch.int32 or packed.shape != expected:
        raise ValueError(
            f"Expected packed INT4 shape/dtype {expected}/int32, "
            f"got {packed.shape}/{packed.dtype}"
        )
    shifts = torch.arange(0, 32, INT4_BITS, device=packed.device, dtype=torch.int32)
    return ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(size_k, size_n).contiguous()


def _pack_cols(values: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    if values.shape != (size_k, size_n):
        raise ValueError(f"Expected unpacked INT4 shape {(size_k, size_n)}, got {values.shape}")
    values = values.to(torch.int32).reshape(size_k, size_n // INT4_PACK_FACTOR, INT4_PACK_FACTOR)
    shifts = torch.arange(0, 32, INT4_BITS, device=values.device, dtype=torch.int32)
    return torch.sum(values << shifts, dim=-1, dtype=torch.int32).contiguous()


def permute_marlin_scales(
    scales: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    expected_groups = 1 if group_size == -1 else size_k // group_size
    if scales.shape != (expected_groups, size_n):
        raise ValueError(
            f"Expected AWQ scales shape {(expected_groups, size_n)}, got {scales.shape}"
        )
    if group_size != -1 and group_size < size_k:
        permutation = [index + 8 * offset for index in range(8) for offset in range(8)]
    else:
        permutation = [
            2 * index + offset
            for index in range(4)
            for offset in (0, 1, 8, 9, 16, 17, 24, 25)
        ]
    return (
        scales.reshape(-1, len(permutation))[:, permutation]
        .reshape(-1, size_n)
        .contiguous()
    )


def convert_awq_zero_points(
    packed_zero_points: torch.Tensor, num_groups: int, size_n: int
) -> torch.Tensor:
    zero_points = _unpack_cols(packed_zero_points, num_groups, size_n)
    undo_interleave = [0, 4, 1, 5, 2, 6, 3, 7]
    zero_points = zero_points.reshape(-1, 8)[:, undo_interleave].reshape(num_groups, size_n)

    scale_permutation = [index + 8 * offset for index in range(8) for offset in range(8)]
    zero_points = zero_points.reshape(-1, 64)[:, scale_permutation]
    interleave = [0, 2, 4, 6, 1, 3, 5, 7]
    zero_points = zero_points.reshape(-1, 8)[:, interleave].reshape(num_groups, size_n)
    return _pack_cols(zero_points, num_groups, size_n)


def prepare_awq_marlin_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    input_size: int,
    output_size: int,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from sgl_kernel import awq_marlin_repack

    if input_size % 128 or output_size % 64:
        raise ValueError(
            "AWQ-Marlin requires input size divisible by 128 and output size divisible by 64"
        )
    resolved_group_size = input_size if group_size == -1 else group_size
    if input_size % resolved_group_size:
        raise ValueError(f"Input size {input_size} is not divisible by group size {group_size}")
    num_groups = input_size // resolved_group_size
    expected_qweight = (input_size, output_size // INT4_PACK_FACTOR)
    expected_qzeros = (num_groups, output_size // INT4_PACK_FACTOR)
    if qweight.dtype != torch.int32 or qweight.shape != expected_qweight:
        raise ValueError(
            f"Expected AWQ qweight shape/dtype {expected_qweight}/int32, "
            f"got {qweight.shape}/{qweight.dtype}"
        )
    if qzeros.dtype != torch.int32 or qzeros.shape != expected_qzeros:
        raise ValueError(
            f"Expected AWQ qzeros shape/dtype {expected_qzeros}/int32, "
            f"got {qzeros.shape}/{qzeros.dtype}"
        )
    if not scales.is_floating_point():
        raise ValueError(f"AWQ scales must be floating point, got {scales.dtype}")

    marlin_weight = awq_marlin_repack(qweight.contiguous(), input_size, output_size, INT4_BITS)
    marlin_scales = permute_marlin_scales(scales, input_size, output_size, group_size)
    marlin_zero_points = convert_awq_zero_points(qzeros, num_groups, output_size)
    workspace_size = max(256, (output_size // 64) * 16)
    workspace = torch.zeros(workspace_size, dtype=torch.int32, device=qweight.device)
    return marlin_weight, marlin_scales, marlin_zero_points, workspace


def awq_marlin_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    workspace: torch.Tensor,
    input_size: int,
    output_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from sgl_kernel import gptq_marlin_gemm
    from sgl_kernel.scalar_type import scalar_types

    x_2d = x.reshape(-1, input_size)
    empty = torch.empty(0, dtype=torch.int32, device=x.device)
    output = gptq_marlin_gemm(
        x_2d,
        None,
        qweight,
        scales,
        None,
        qzeros,
        empty,
        empty,
        workspace,
        scalar_types.uint4,
        x_2d.shape[0],
        output_size,
        input_size,
        is_k_full=True,
        use_fp32_reduce=True,
    )
    if bias is not None:
        output.add_(bias)
    return output.reshape(*x.shape[:-1], output_size)
