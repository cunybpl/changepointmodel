""" These are wrappers around sklearn.metrics interface. They can interop with sklearn scorers or 
can be used on their own.
"""

import abc
from dataclasses import dataclass
from .nptypes import AnyByAnyNDArrayField, OneDimNDArray
from typing import Any, Callable, List, TypeVar, Union
from .calc import metrics
from .estimator import EnergyChangepointEstimator


class IComparable(
    abc.ABC
):  # trick to declare a Comparable type... py3 all comparability is implemented in terms of < so this is a safe descriptor
    @abc.abstractmethod
    def __lt__(self, other: Any) -> bool:
        ...


ComparableType = TypeVar("ComparableType", bound=IComparable)
SklScoreReturnType = Union[float, AnyByAnyNDArrayField, Any]
Comparator = Callable[[IComparable, IComparable], bool]


@dataclass
class Score(object):
    name: str
    value: SklScoreReturnType
    threshold: float
    ok: bool


class IEval(abc.ABC):
    @abc.abstractmethod
    def ok(self, estimator: EnergyChangepointEstimator) -> bool:
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
        self, y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs
    ) -> SklScoreReturnType:
        ...

    def __call__(self, *args, **kwargs):
        return self.calc(*args, **kwargs)


class R2(ScoringFunction):
    def calc(
        self, y: OneDimNDArray, pred_y: OneDimNDArray, **kwargs
    ) -> SklScoreReturnType:
        return metrics.r2_score(y, pred_y, **kwargs)


class Rmse(ScoringFunction):
    def calc(
        self, y: OneDimNDArray, pred_y: OneDimNDArray, **kwargs
    ) -> SklScoreReturnType:
        return metrics.rmse(y, pred_y, **kwargs)


class Cvrmse(ScoringFunction):
    def calc(
        self, y: OneDimNDArray, pred_y: OneDimNDArray, **kwargs
    ) -> SklScoreReturnType:
        return metrics.cvrmse(y, pred_y, **kwargs)


class ScoreEval(IEval):
    def __init__(self, scorer: ScoringFunction, threshold: float, method: Comparator):
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

    def ok(self, estimator: EnergyChangepointEstimator) -> Score:
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

    def check(self, estimator: EnergyChangepointEstimator) -> List[Score]:
        """Given a changepoint estimator return a list of score evaluations against the result.

        Args:
            estimator (EnergyChangepointEstimator): _description_

        Returns:
            List[Score]: _description_
        """
        return [e.ok(estimator) for e in self._evals]
