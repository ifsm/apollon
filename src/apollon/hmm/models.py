# pylint: disable = C0114, C0115, R0903

from typing import Self

from chainsaddiction.poishmm import PoisHmm
import numpy as _np

from pydantic import BaseModel
from apollon.types import FloatArray


class PoissonHmmParams(BaseModel):
    lambda_: FloatArray
    gamma_: FloatArray
    delta_: FloatArray

    @classmethod
    def from_result(cls, obj: PoisHmm) -> Self:
        """Parse PoissonHmmParams from chainsaddiction.PoisHmm

        Args:
            obj:    PoisHmm result object

        Returns:
            PoissonHmmParams
        """
        return cls(lambda_=obj.lambda_.astype(_np.float64),
                   gamma_=obj.gamma_.astype(_np.float64),
                   delta_=obj.delta_.astype(_np.float64))


class PoissonHmmQualityMeasures(BaseModel):
    aic: float
    bic: float
    nllk: float
    n_iter: float

    @classmethod
    def from_result(cls, obj: PoisHmm) -> Self:
        """Parse quality measure from chainsaddiction.PoisHmm

        Args:
            obj:    PoisHmm result object

        Returns:
            Quality measures data model
        """
        return cls(aic=obj.aic, bic=obj.bic, nllk=obj.llk,
                   n_iter=obj.n_iter)
