"""
=======================================
Onset detection algorithms
=======================================

.. currentmodule:: apollon.onsets

Apollon prodvides a number of different algorithms for onset detection which
are suitable given different characteristics of the signal under consideration.

Algorithms
-----------

.. autosummary::
    :toctree: generated/

    apollon.onsets.OnsetDetector            Base class
    apollon.onsets.FluxOnsetDetector        OB based on spectral flux
    apollon.onsets.EntropyOnsetDetector     OD based on signal entropy


Evaluation helpers
-------------------

.. autosummary::
    :toctree: generated/

    apollon.onsets.evaluate_onsets      Evaluate detector result given ground truth.


Data models
------------

Each onset detector has its own parameter model. These models define the types
and value ranges for each parameter. They can also be used for easy
serialization of the input parameters. To this end, each model implements the
methods :code:`model_dump`, and :code:`model_dump_json`, which return the model
data as plain Python dictionary or JSON string, respectively.

.. autosummary::
    :toctree: generated/

    apollon.onsets.models.OnsetDetectorParams
    apollon.onsets.models.FluxODParams
    apollon.onsets.models.EntropyODParams

"""

from . detectors import OnsetDetector, FluxOnsetDetector, EntropyOnsetDetector
from . import models
from . _eval import evaluate_onsets

__all__ = ("evaluate_onsets", "models", "OnsetDetector", "FluxOnsetDetector",
           "EntropyOnsetDetector")
