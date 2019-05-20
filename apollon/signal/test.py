import numpy as np
from apollon.audio import loadwav
from apollon.signal.tools import trim_spectrogram
from apollon.signal.features import spectral_shape
from apollon.signal.spectral import stft
import matplotlib.pyplot as plt

s = loadwav('/Users/michael/audio/D-Dorisch.wav')
S = stft(s.data, s.fs, n_perseg=2048, hop_size=1024)

S.plot('plasma', log_frq=1000, low=200, high=5000)
plt.show()
