""" This module provides filtering functionality for the application. In many cases when a model fails to 
meet certain statistical or logical criteria we do not want it returned. 

By using this module we can analyze and filter the results after they have been generated.


"""

from typing import List
from .base import BemaChangepointResultContainers, BemaChangepointResultContainer
from .models import FilterHowEnum, FilterWhichEnum
from . import extras

from changepointmodel.core.pmodels import ParamaterModelCallableT, EnergyParameterModelT
from typing import Generic


class _MungedResult(Generic[ParamaterModelCallableT, EnergyParameterModelT]):
    def __init__(
        self,
        result: BemaChangepointResultContainer[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
        score_name: str,
    ):
        self.result: BemaChangepointResultContainer[
            ParamaterModelCallableT, EnergyParameterModelT
        ] = result
        self.score_name = score_name
        self.score = [s for s in result.result.scores if s.name == score_name].pop()

    def is_score_better_then(self, other: float) -> bool:
        if self.score_name == "r2":
            return True if self.score.value > other else False
        elif self.score_name == "cvrmse":
            return True if self.score.value < other else False
        else:
            raise ValueError("Unsupported score type")


MungedResults = List[_MungedResult[ParamaterModelCallableT, EnergyParameterModelT]]


def _filter_best_score(
    munged: MungedResults[ParamaterModelCallableT, EnergyParameterModelT],
) -> BemaChangepointResultContainers[ParamaterModelCallableT, EnergyParameterModelT]:
    if len(munged) == 1:
        return [munged[0].result]

    current_best = munged[0]  # start with the first
    for m in munged[1:]:
        if m.is_score_better_then(
            float(current_best.score.value)
        ):  # NOTE that a tie will go to first in list.. this would be an extremely rare edge case probably only w/ bad data.
            current_best = m
    return [current_best.result]


def _filter_threshold_ok(
    munged: MungedResults[ParamaterModelCallableT, EnergyParameterModelT],
) -> BemaChangepointResultContainers[ParamaterModelCallableT, EnergyParameterModelT]:
    return [m.result for m in munged if m.score.ok]


def _filter_threshold_ok_first_is_best(
    munged: MungedResults[ParamaterModelCallableT, EnergyParameterModelT],
) -> BemaChangepointResultContainers[ParamaterModelCallableT, EnergyParameterModelT]:
    ok = [m.result for m in munged if m.score.ok]
    if len(ok) == 0:
        return ok
    return [ok[0]]


class ChangepointEstimatorFilter(object):
    def __init__(self, which: FilterWhichEnum, how: FilterHowEnum, extras: bool = True):
        """This object is responsible for filtering models after they have been fit using a set
        of criteria at runtime.

        The provided parameters:
            1. which - will filter either on r2 or cvrmse depending on your use case. Higher r2 values are better
                and lower cvrmse values are better.
            2. how - the process used to filter.
                * `best_score` - simply picks the best r2 or cvrmse based on your choice in `which`. This does not need
                    to meet the threshold criteria.
                * `threshold_ok` - filters on whether your chosen score is above the threshold. This will yield all results
                    that meet the threshold criteria and drop those that do not.
                * `threshold_ok_first_is_best` - filters on whether your chosen score is above the threshold. From
                those results it will return the top result in the list.
            3. extras - runs a gauntlet of extra checks described in the `extras` module.

        Args:
            which (FilterWhichEnum): Filter on either r2 or cvrmse.
            how (FilterHowEnum): How to filter the model.
            extras (bool, optional): Use functionality in the extras module to further filter models using data population,
                shape, and tstat tests. Defaults to True.
        """
        self._which = which
        self._how = how
        self._extras = extras

    def filtered(
        self,
        results: BemaChangepointResultContainers[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> List[
        BemaChangepointResultContainer[ParamaterModelCallableT, EnergyParameterModelT]
    ]:
        """Filters these results based on the provided parameters returning 0:N result containers based on the
        configuration.

        NOTE this is refactored behavior from # 212, 213

        Args:
            results (BemaChangepointResultContainers): _description_

        Returns:
            BemaChangepointResultContainer: A filtered changepointmodel result container.
        """
        if self._extras:
            results = [
                r for r in extras.shape(results)
            ]  # filter out bad shapes of models
            results = [
                r for r in extras.dpop(results)
            ]  # check heatmonths coolmonths data population
            results = [r for r in extras.tstat(results)]  # check tstat...
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
