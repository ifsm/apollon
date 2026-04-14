# pylint: disable = C0114, C0115, R0903

from typing import Self
from pydantic import BaseModel, model_validator


class SpectralTransformParams(BaseModel):
    fps: int
    window: str | None = None
    n_fft: int | None = None


class DftParams(SpectralTransformParams):
    norm: bool = True


class StftParams(DftParams):
    n_perseg: int
    n_overlap: int
    extend: bool
    pad: bool


class CorrDimParams(BaseModel):
    delay: int
    m_dim: int
    n_bins: int
    scaling_size: int


class CorrGramParams(BaseModel):
    wlen: int
    n_delay: int
    total: bool = True


class TriangFilterSpec(BaseModel):
    low: float
    high: float
    n_filters: int

    @model_validator(mode="after")
    def _check_low_lt_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low freq must be less then high")
        return self
