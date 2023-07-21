"""
=======================================
Onset detection algorithms
=======================================

.. currentmodule:: apollon.onsets


.. autosummary::
    :toctree: generated/
    :nosignatures:

    OnsetDetector            Base class
    FluxOnsetDetector        OB based on spectral flux
    EntropyOnsetDetector     OD based on signal entropy
    evaluate_onsets          Evaluate detector result given ground truth.
    models.OnsetDetectorParams
    models.FluxODParams
    models.EntropyODParams
"""

from . detectors import OnsetDetector, FluxOnsetDetector, EntropyOnsetDetector
from . import models
from . _eval import evaluate_onsets

__all__ = ("evaluate_onsets", "models", "OnsetDetector", "FluxOnsetDetector",
           "EntropyOnsetDetector")
