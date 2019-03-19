import numpy as _np
from scipy.signal import hilbert as _hilbert

from .. import segment as _segment
from .. types import Array as _Array


def log_attack_time(inp: _Array, fs: int, ons_idx: _Array, wlen:float=0.05) -> _Array:
    """Estimate the attack time of each onset an return is logarithm.

    This function estimates the attack time as the duration between the
    onset and the local maxima of the magnitude of the Hilbert transform
    of the local window.

    Args:
        inp     (ndarray)    Input signal.
        fs      (int)        Sampling frequency.
        ons_idx (ndarray)    Sample indices of onsets.
        wlen    (float)      Local window length in samples.

    Returns:
        (ndarray)    Logarithm of the attack time.
    """
    wlen = int(fs * wlen)
    segs = _segment.by_onsets(inp, wlen, ons_idx)
    mx = _np.absolute(_hilbert(segs)).argmax(axis=1)

    return _np.log(mx/fs)
