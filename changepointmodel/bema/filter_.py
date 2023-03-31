import logging
from typing import List
from .base import BemaChangepointResultContainers, BemaChangepointResultContainer
from .models import FilterHowEnum, FilterWhichEnum
from . import extras


class _MungedResult(object):
    def __init__(self, result: BemaChangepointResultContainer, score_name: str):
        self.result = result
        self.score_name = score_name
        self.score = [s for s in result.result.scores if s.name == score_name].pop()

    def is_score_better_then(self, other: float):
        if self.score_name == "r2":
            return True if self.score.value > other else False
        elif self.score_name == "cvrmse":
            return True if self.score.value < other else False


MungedResults = List[_MungedResult]


def _filter_best_score(munged: MungedResults) -> BemaChangepointResultContainers:
    if len(munged) == 1:
        return [munged[0].result]

    current_best = munged[0]  # start with the first
    for m in munged[1:]:
        if m.is_score_better_then(
            current_best.score.value
        ):  # NOTE that a tie will go to first in list.. this would be an extremely rare edge case probably only w/ bad data.
            current_best = m
    return [current_best.result]


def _filter_threshold_ok(munged: MungedResults) -> BemaChangepointResultContainers:
    return [m.result for m in munged if m.score.ok]


def _filter_threshold_ok_first_is_best(
    munged: MungedResults,
) -> BemaChangepointResultContainers:
    ok = [m.result for m in munged if m.score.ok]
    if len(ok) == 0:
        return ok
    return [ok[0]]


class ChangepointEstimatorFilter(object):
    def __init__(self, which: FilterWhichEnum, how: FilterHowEnum, extras: bool = True):
        self._which = which
        self._how = how
        self._extras = extras

    def filtered(self, results: BemaChangepointResultContainers) -> MungedResults:
        """Fitler models and return MungedResults.

        NOTE this is refactored behavior from # 212, 213

        Args:
            results (BemaChangepointResultContainers): _description_

        Raises:
            ValueError: _description_

        Returns:
            MungedResults: _description_
        """
        if self._extras:
            results = [
                r for r in extras.shape(results)
            ]  # filter out bad shapes of models
            results = [
                r for r in extras.dpop(results)
            ]  # check heatmonths coolmonths data population
            results = [
                r for r in extras.tstat(results)
            ]  # check tstat... i guess its supposed to do something...
            if len(results) == 0:
                return results

        munged = [_MungedResult(result, self._which.value) for result in results]

        if self._how.value == "best_score":  # will always return 1
            return _filter_best_score(munged)

        elif (
            self._how.value == "threshold_ok"
        ):  # returns 0:N for scores above threshold
            return _filter_threshold_ok(munged)

        elif self._how.value == "threshold_ok_first_is_best":  # returns 1 or None
            return _filter_threshold_ok_first_is_best(munged)

        else:
            raise ValueError(
                "Unsupported enum in _filter_estimators"
            )  # should not reach but defaults in switch blocks are good.
