""" These are wrappers around sklearn.metrics interface. They can interop with sklearn scorers or 
can be used on their own.
"""

import abc
from dataclasses import dataclass
from .nptypes import AnyByAnyNDArrayField, OneDimNDArray
from typing import Any, Callable, List, TypeVar, Union
from .calc import metrics
from .estimator import EnergyChangepointEstimator
from .pmodels import ParamaterModelCallableT, EnergyParameterModelT

import numpy as np
from typing import Dict, Any


SklScoreReturnType = Union[float, OneDimNDArray[np.float64]]


class ISklComparable(abc.ABC):
    @abc.abstractmethod
    def __call__(self, result: SklScoreReturnType, threshold: float) -> bool:
        ...


@dataclass
class Score(object):
    name: str
    value: SklScoreReturnType
    threshold: float
    ok: bool


class IEval(abc.ABC):
    @abc.abstractmethod
    def ok(
        self,
        estimator: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> Score:
        ...


class ScoringFunction(abc.ABC):
    """Interface for a scoring function. User should override calc with a numpy/Cython based
    function that takes (y, pred_y) and returns a scalar. A name should be provided so the object
    deserializations.
    """

    @property
    def name(self) -> str:
        """Expose the name of the class conveniently for creating collections of scores.

        Returns:
            str: [description]
        """
        return self.__class__.__name__.lower()

    @abc.abstractmethod
    def calc(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
        **kwargs: Dict[str, Any]
    ) -> SklScoreReturnType:
        ...

    def __call__(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
        **kwargs: Dict[str, Any]
    ) -> SklScoreReturnType:
        return self.calc(y, y_pred, **kwargs)


class R2(ScoringFunction):
    def calc(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
        **kwargs: Dict[str, Any]
    ) -> SklScoreReturnType:
        return metrics.r2_score(y, y_pred, **kwargs)


class Rmse(ScoringFunction):
    def calc(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
        **kwargs: Dict[str, Any]
    ) -> SklScoreReturnType:
        return metrics.rmse(y, y_pred, **kwargs)


class Cvrmse(ScoringFunction):
    def calc(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
        **kwargs: Dict[str, Any]
    ) -> SklScoreReturnType:
        return metrics.cvrmse(y, y_pred, **kwargs)


class ScoreEval(IEval):
    def __init__(
        self,
        scorer: ScoringFunction,
        threshold: float,
        method: ISklComparable,
    ):
        """Evaluates a single score using the scoring function and reports whether the Score meets the
        threshold.

        Args:
            scorer (ScoringFunction): _description_
            threshold (float): _description_
            method (Comparator): _description_
        """
        self._scorer = scorer
        self._threshold = threshold
        self._method = method

    def ok(
        self,
        estimator: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> Score:
        est_val = self._scorer(estimator.y, estimator.pred_y)
        return Score(
            self._scorer.name,
            est_val,
            self._threshold,
            self._method(est_val, self._threshold),
        )


class Scorer(object):
    def __init__(self, evals: List[ScoreEval]):
        """Executes a series of ScoreEvals.

        Args:
            evals (List[ScoreEval]): _description_
        """
        self._evals = evals

    def check(
        self,
        estimator: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> List[Score]:
        """Given a changepoint estimator return a list of score evaluations against the result.

        Args:
            estimator (EnergyChangepointEstimator): _description_

        Returns:
            List[Score]: _description_
        """
        return [e.ok(estimator) for e in self._evals]
