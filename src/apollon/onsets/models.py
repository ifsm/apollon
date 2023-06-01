from pydantic import BaseModel


class FluxODParams(BaseModel):
    fps: int
    window: str
    n_perseg: int
    n_overlap: int


class EntropyODParams(BaseModel):
    fps: int
    m_dim: int
    delay: int
    bins: int
    n_perseg: int
    n_overlap: int
