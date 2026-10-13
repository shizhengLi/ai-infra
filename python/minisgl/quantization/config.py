from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


QuantizationMethod = Literal["fp8", "awq"]


@dataclass(frozen=True)
class QuantizationConfig:
    method: QuantizationMethod
    bits: int = 8
    group_size: int = -1
    zero_point: bool = False

    @classmethod
    def from_name(
        cls, name: str | None, checkpoint_config: dict[str, Any] | None = None
    ) -> QuantizationConfig | None:
        if name is None:
            return None
        if name == "fp8":
            return cls(method="fp8")
        if name != "awq":
            raise ValueError(f"Unsupported quantization method: {name}")
        if checkpoint_config is None:
            raise ValueError("AWQ requires quantization_config metadata in the checkpoint")

        bits = checkpoint_config.get("bits")
        group_size = checkpoint_config.get("group_size")
        quant_method = checkpoint_config.get("quant_method")
        version = str(checkpoint_config.get("version", "")).lower()
        zero_point = checkpoint_config.get("zero_point")
        if (
            quant_method != "awq"
            or bits != 4
            or group_size not in (-1, 32, 64, 128)
            or version != "gemm"
            or zero_point is not True
        ):
            raise ValueError(
                "AWQ-Marlin requires 4-bit GEMM weights with group size -1/32/64/128 "
                "and zero points"
            )
        return cls(method="awq", bits=bits, group_size=group_size, zero_point=True)
