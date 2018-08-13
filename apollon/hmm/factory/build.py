#! python3

from .. poisson_hmm import PoissonHmm as _PoissonHmm
from ... signal.audio import loadwav
from ... onsets import FluxOnsetDetector2


def PoissonHmm(path: str, params: dict = None):
    """Train HMM on single audio file.

    Params:
        path    (str)   Path to audio file.
        params  (dict)  dict of modelling parameters.
    """
    x = loadwav(path)
    ons = FluxOnsetDetector2()
    
