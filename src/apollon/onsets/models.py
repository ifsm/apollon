# pylint: disable = C0114, C0115, R0903

from pydantic import BaseModel


class OnsetDetectorParams(BaseModel):
    fps: int
    n_perseg: int
    n_overlap: int


class FluxODParams(OnsetDetectorParams):
    window: str


class EntropyODParams(OnsetDetectorParams):
    m_dim: int
    delay: int
    bins: int
