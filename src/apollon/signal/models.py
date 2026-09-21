# pylint: disable = C0114, C0115, R0903

from typing import Self, Literal
from pydantic import BaseModel, model_validator, NonNegativeFloat, PositiveInt


Normalization = Literal["amplitude", "ortho"]
"""Scaling convention of a spectral transform, see ``spectral.fft``."""


class SpectralTransformParams(BaseModel):
    fps: int
    window: str | None = None
    n_fft: int | None = None


class DftParams(SpectralTransformParams):
    norm: Normalization | None = "amplitude"
    single_sided: bool = True


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
    scale: Literal["mel", "hz"] = "mel"

    @model_validator(mode="after")
    def _check_low_lt_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low freq must be less then high")
        return self


class CepstrumParams(BaseModel):
    n_coefs: PositiveInt = 13
    dct_type: Literal[1, 2, 3, 4] = 2
    lifter_gain: float = 24.0


class CepstralParams(BaseModel):
    fb: TriangFilterSpec
    top_db: NonNegativeFloat | None = None
    cepstrum: CepstrumParams = CepstrumParams()

    @model_validator(mode="after")
    def _check_n_coefs_le_n_filters(self) -> Self:
        if self.cepstrum.n_coefs > self.fb.n_filters:
            raise ValueError(f"Requested {self.cepstrum.n_coefs} cepstral "
                             f"coefficients from {self.fb.n_filters} filters. "
                             "The cepstrum cannot hold more coefficients than "
                             "the filter bank has filters.")
        return self


class MfccParams(CepstralParams):
    stft: StftParams
    preemphasis: float | None = 0.97
