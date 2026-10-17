from __future__ import annotations

import json
from pathlib import Path

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


FP8_KV_DTYPE = torch.float8_e4m3fn
FP8_KV_MAX = 448.0
FP8_KV_SCALE_MARGIN = 0.95


def calibrate_fp8_kv_scale(k: torch.Tensor, v: torch.Tensor) -> float:
    """Estimate a scalar dequantization scale from one layer's K/V activations."""
    max_abs = torch.maximum(k.float().abs().amax(), v.float().abs().amax()).item()
    return max(float(max_abs) / (FP8_KV_MAX * FP8_KV_SCALE_MARGIN), 1e-6)


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        scale: float = 1.0,
        calibration: str = "fixed",
        calibration_summary: str | None = None,
        fused_store: bool = False,
        native_pair: bool = False,
    ) -> None:
        if scale <= 0.0:
            raise ValueError(f"KV cache scale must be positive, got {scale}")
        if calibration not in {"fixed", "first_batch"}:
            raise ValueError(f"Unsupported KV cache calibration: {calibration}")
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._scale = float(scale)
        self._calibration = calibration
        self._layer_scales = [self._scale] * num_layers
        self._layer_calibrated = [calibration == "fixed"] * num_layers
        self._layer_scale_tensors = (
            [torch.full((1,), self._scale, device=device, dtype=torch.float32) for _ in range(num_layers)]
            if dtype == FP8_KV_DTYPE
            else []
        )
        self._calibration_summary_path = calibration_summary
        self._fused_store = fused_store
        self._native_pair = native_pair
        self._layer_written_elements = [0] * num_layers
        self._layer_saturated_elements = [0] * num_layers
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        if self.dtype == FP8_KV_DTYPE:
            scale = self._layer_scales[layer_id]
            if not self._layer_calibrated[layer_id]:
                # CUDA Graph capture must not call .item() on a captured tensor. The
                # eager prefill path is the intended calibration path; capture falls
                # back to the explicit CLI scale and remains deterministic.
                is_capturing = k.is_cuda and torch.cuda.is_current_stream_capturing()
                if not is_capturing:
                    scale = calibrate_fp8_kv_scale(k, v)
                    self._layer_scales[layer_id] = scale
                    self._layer_calibrated[layer_id] = True
                    self._layer_scale_tensors[layer_id].fill_(scale)
            # FlashInfer consumes the matching dequantization scale at run time.
            if self._calibration_summary_path is not None:
                saturated = (
                    (k.float().abs() >= scale * FP8_KV_MAX).sum()
                    + (v.float().abs() >= scale * FP8_KV_MAX).sum()
                )
                self._layer_written_elements[layer_id] += 2 * k.numel()
                self._layer_saturated_elements[layer_id] += int(saturated.item())
            if self._fused_store:
                from minisgl.kernel import store_fp8_cache

                store_fp8_cache(
                    k_cache=self._k_buffer[layer_id].view(self._storage_shape),
                    v_cache=self._v_buffer[layer_id].view(self._storage_shape),
                    indices=out_loc,
                    k=k,
                    v=v,
                    scale=self._layer_scale_tensors[layer_id],
                    native_pair=self._native_pair,
                )
                return
            k = (k.float() / scale).clamp(-FP8_KV_MAX, FP8_KV_MAX).to(self.dtype)
            v = (v.float() / scale).clamp(-FP8_KV_MAX, FP8_KV_MAX).to(self.dtype)
        elif k.dtype != self.dtype or v.dtype != self.dtype:
            k = k.to(self.dtype)
            v = v.to(self.dtype)

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def scale(self) -> float:
        return self._scale

    def get_scale(self, layer_id: int) -> float:
        return self._layer_scales[layer_id]

    def write_calibration_summary(self, path: str, rank: int) -> None:
        if self._calibration_summary_path is None:
            return
        target = Path(path)
        suffix = target.suffix or ".json"
        target = target.with_name(f"{target.stem}.rank{rank}{suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        layers = []
        for layer_id, scale in enumerate(self._layer_scales):
            written = self._layer_written_elements[layer_id]
            saturated = self._layer_saturated_elements[layer_id]
            layers.append(
                {
                    "layer_id": layer_id,
                    "scale": scale,
                    "calibrated": self._layer_calibrated[layer_id],
                    "written_elements": written,
                    "saturated_elements": saturated,
                    "saturation_fraction": saturated / written if written else 0.0,
                }
            )
        target.write_text(
            json.dumps(
                {
                    "rank": rank,
                    "dtype": str(self.dtype),
                    "calibration": self._calibration,
                    "layers": layers,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers
