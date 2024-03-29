""" A Helper class that given an estimator will predict the usage of some X and return the sum of the predictions.
This is useful for calculating NormalizedAnnualConsumption.
"""

from .nptypes import NByOneNDArray
from .estimator import EnergyChangepointEstimator
from dataclasses import dataclass
import numpy as np
from .pmodels import ParamaterModelCallableT, EnergyParameterModelT

import warnings

_deprec = """
The predsum module is deprecrecated since 3.3 and may eventually be removed. 
The same calculation can be performed directly on the estimator using `nac` method.
>>> estimator.nac()
or 
>>> estimator.nac(scalar=30.437)  # to scale out avg_day values to monthly gross
"""


@dataclass
class PredictedSum(object):
    value: float


class PredictedSumCalculator(object):
    def __init__(self, X: NByOneNDArray[np.float64]):
        """The array of X data you wish to predict and summarize given an estimator.

        Args:
            X (NByOneNDArray): A 1d array of data to be predicted on.
        """
        self._X = X
        warnings.warn(_deprec, DeprecationWarning, stacklevel=2)

    def calculate(
        self,
        estimator: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> PredictedSum:
        """Given the estimator will predict on this object's X and return its sum.

        Args:
            estimator (EnergyChangepointEstimator): A fitted EnergyChangepointEstimator.

        Returns:
            PredictedSum: The sum of predictions
        """
        return PredictedSum(value=np.sum(estimator.predict(self._X)))  # type: ignore
