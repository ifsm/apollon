# pylint: disable = C0114, C0115, R0903

from pydantic import BaseModel


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
