import numpy as np
from apollon.datasets import load_earthquakes

x = load_earthquakes().data

def params(x, m, mv):
    lam = np.linspace(x.min(), x.max(), m, endpoint=True)
    gam = np.empty((m, m))
    gam.fill((1-mv) / (m-1))
    fill_diagonal(gam, mv)
    delta = np.repeat(1/m, m)
    return (lam, gam, delta)

