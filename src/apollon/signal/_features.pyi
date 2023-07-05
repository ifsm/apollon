from apollon.types import NDArray, FloatArray

def correlogram(inp: NDArray, wlen: int, delay_max: int) -> FloatArray:
    ...

def correlogram_delay(inp: NDArray, delays: NDArray,
                      off_max: int) -> FloatArray:
    ...

def emb_dists(inp: NDArray, delay: int, m_dim: int) -> FloatArray:
    ...

def cdim_bader(inp: NDArray, delay: int, m_dim: int, n_bins: int,
               scaling_size: int) -> float:
    ...
