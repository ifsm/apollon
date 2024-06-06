from apollon.types import Array, FloatArray

def correlogram(inp: Array, wlen: int, delay_max: int) -> FloatArray:
    ...

def correlogram_delay(inp: Array, delays: Array,
                      off_max: int) -> FloatArray:
    ...

def emb_dists(inp: Array, delay: int, m_dim: int) -> FloatArray:
    ...

def cdim_bader(inp: Array, delay: int, m_dim: int, n_bins: int,
               scaling_size: int) -> float:
    ...
