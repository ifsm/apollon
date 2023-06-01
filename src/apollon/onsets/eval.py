"""
Evaluation helpers
"""

import numpy as np
from apollon.types import FloatArray


def evaluate_onsets(targets: dict[str, FloatArray],
                    estimates: dict[str, FloatArray]
                    ) -> tuple[float, float, float]:
    """Evaluate onset detection performance.

    This function uses the mir_eval package for evaluation.

    Params:
        targets:    Ground truth onset times, with dict keys being file names,
                    and values being target onset time codes in ms.

        estimates:  Estimated onsets times, with dictkeys being file names,
                    and values being the estimated onset time codes in ms.

    Returns:
        Precison, recall, f-measure.
    """
    out = []
    for name, tvals in targets.items():
        od_eval = _me.onset.evaluate(tvals, estimates[name])
        out.append([i for i in od_eval.values()])

    return np.array(out)
