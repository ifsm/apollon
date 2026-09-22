# pylint: disable = C0114, C0115, R0903

from typing import Self, Literal
from pydantic import (BaseModel, FiniteFloat, model_validator,
                      NonNegativeFloat, NonNegativeInt, PositiveInt)


Normalization = Literal["amplitude", "ortho"]
"""Scaling convention of a spectral transform, see ``spectral.fft``."""


class SpectralTransformParams(BaseModel):
    fps: PositiveInt
    window: str | None = None
    n_fft: PositiveInt | None = None


class DftParams(SpectralTransformParams):
    norm: Normalization | None = "amplitude"
    single_sided: bool = True


class StftParams(DftParams):
    n_perseg: PositiveInt
    n_overlap: NonNegativeInt
    extend: bool
    pad: bool

    @model_validator(mode="after")
    def _check_segment_lengths(self) -> Self:
        if self.n_overlap >= self.n_perseg:
            raise ValueError(f"n_overlap ({self.n_overlap}) must be less "
                             f"than n_perseg ({self.n_perseg}).")
        if self.n_fft is not None and self.n_fft < self.n_perseg:
            raise ValueError(f"n_fft ({self.n_fft}) must not be less than "
                             f"n_perseg ({self.n_perseg}); a shorter FFT "
                             "would crop every segment.")
        return self


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
    low: NonNegativeFloat
    high: float
    n_filters: PositiveInt
    scale: Literal["mel", "hz"] = "mel"

    @model_validator(mode="after")
    def _check_low_lt_high(self) -> Self:
        if self.low >= self.high:
            raise ValueError("low freq must be less then high")
        return self


class CepstrumParams(BaseModel):
    n_coefs: PositiveInt = 13
    dct_type: Literal[1, 2, 3, 4] = 2
    lifter_gain: NonNegativeFloat = 24.0


class CepstralParams(BaseModel):
    fb: TriangFilterSpec
    floor_dbfs: FiniteFloat = -100.0
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
