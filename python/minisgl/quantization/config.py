from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


QuantizationMethod = Literal["fp8"]


@dataclass(frozen=True)
class QuantizationConfig:
    method: QuantizationMethod

    @classmethod
    def from_name(cls, name: str | None) -> QuantizationConfig | None:
        if name is None:
            return None
        if name != "fp8":
            raise ValueError(f"Unsupported quantization method: {name}")
        return cls(method="fp8")
