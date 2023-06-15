import numpy as np
from . types import FloatArray


class FilterPeakPicker:
    def __init__(self, n_after: int = 10, n_before: int = 10,
                 alpha: float = .1, delta: float=.1) -> None:
        self.n_after = n_after
        self.n_before = n_before
        self.alpha = alpha
        self.delta = delta

    def detect(self, inp: FloatArray) -> FloatArray:
        """Pick local maxima from a numerical time series.

        Pick local maxima from the onset detection function `odf`, which is assumed
        to be an one-dimensional array. Typically, `odf` is the Spectral Flux per
        time step.

        Args:
            odf:         Onset detection function, e.g., Spectral Flux.
            n_after: Window lenght to consider after now.
            n_before:  Window lenght to consider before now.
            alpha:       Smoothing factor. Must be in ]0, 1[.
            delta:       Difference to the mean.

        Return:
            Peak indices.
        """
        g = [0]
        out = []

        for n, val in enumerate(inp):
            # set local window
            idx = np.arange(n-self.n_before, n+self.n_after+1, 1)
            window = np.take(inp, idx, mode='clip')

            cond1 = np.all(val >= window)
            cond2 = val >= (np.mean(window) + self.delta)

            foo = max(val, self.alpha*g[n] + (1-self.alpha)*val)
            g.append(foo)
            cond3 = val >= foo

            if cond1 and cond2 and cond3:
                out.append(n)

        return np.array(out)
