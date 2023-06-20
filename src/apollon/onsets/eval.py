"""
Evaluation helpers
"""

import mir_eval as _me
from apollon.types import FloatArray


def evaluate_onsets(targets: FloatArray,
                    estimates: FloatArray
                    ) -> tuple[float, float, float]:
    """Evaluate onset detection performance

    Args:
        targets:    Ground truth onset times
        estimates:  Estimated onsets times

    Returns:
        Precison, recall, f-measure
    """
    res: tuple[float, float, float]
    res = _me.onset.evaluate(targets, estimates)
    return res
