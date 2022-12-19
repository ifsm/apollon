from pydantic import BaseModel


class DftParams(BaseModel):
    fps: int
    window: str = 'hamming'
    n_fft: int | None = None


class StftParams(DftParams):
    n_perseg: int | None= None
    n_overlap: int | None = None
    extend: bool | None = None
    pad: bool | None = None


class CorrDimParams(BaseModel):
    delay: int
    m_dim: int
    n_bins: int
    scaling_size: int


class CorrGramParams(BaseModel):
    wlen: int
    n_delay: int
    total: bool = True
