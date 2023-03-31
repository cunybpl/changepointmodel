from typing import List, Optional
import numpy as np

from changepointmodel.core import nptypes, scoring
from changepointmodel.core.utils import argsort_1d_idx
from changepointmodel.core.schemas import CurvefitEstimatorDataModel
from changepointmodel.core.predsum import PredictedSumCalculator
from changepointmodel.core.estimator import EnergyChangepointEstimator
from changepointmodel.core.savings import (
    AshraeAdjustedSavingsCalculator,
    AshraeNormalizedSavingsCalculator,
)

from . import config
from .filter_ import ChangepointEstimatorFilter
from .results import BemaChangepointResult, BemaSavingsResult

from .base import BemaChangepointResultContainer, BemaChangepointResultContainers

from .models import (
    BaselineChangepointModelRequest,
    EnergyChangepointModelResponse,
    FilterConfig,
    SavingsRequest,
    SavingsResponse,
)

import logging

logger = logging.getLogger(__name__)

from .exc import BemaChangepointException, bema_changepoint_exception_wrapper


class BemaChangepointModeler(object):
    def __init__(
        self,
        oat: List[float],
        usage: List[float],
        models: List[str] = None,
        r2_threshold: float = 0.75,
        cvrmse_threshold: float = 0.5,
        norms: List[float] = None,
        estimator_filter: ChangepointEstimatorFilter = None,
    ):
        """Runs a single set of changepoint models using the API from the `changepointmodel` lib.

        Args:
            oat (List[float]): _description_
            usage (List[float]): _description_
            models (List[str], optional): _description_. Defaults to None.
            r2_threshold (float, optional): _description_. Defaults to 0.75.
            cvrmse_threshold (float, optional): _description_. Defaults to 0.5.
            norms (List[float], optional): _description_. Defaults to None.
            estimator_filter (ChangepointEstimatorFilter, optional): _description_. Defaults to None.
        """

        self._input_data = CurvefitEstimatorDataModel(X=oat, y=usage)
        self._models = (
            models if models is not None else ["5P", "4P", "3PC", "3PH", "2P"]
        )
        self._scorer = self._make_scorer(r2_threshold, cvrmse_threshold)
        self._nac_calc = self._prep_nac(norms)
        self._estimator_filter = estimator_filter

    def _make_scorer(
        self, r2_threshold: float, cvrmse_threshold: float
    ) -> scoring.Scorer:
        r2_scorer = scoring.ScoreEval(scoring.R2(), r2_threshold, lambda a, b: a > b)
        cvrmse_scorer = scoring.ScoreEval(
            scoring.Cvrmse(), cvrmse_threshold, lambda a, b: a < b
        )

        return scoring.Scorer([r2_scorer, cvrmse_scorer])

    def _fit(self, estimator: EnergyChangepointEstimator):
        X, y, original_ordering = argsort_1d_idx(self._input_data.X, self._input_data.y)
        estimator.fit(X, y)
        estimator._original_ordering = original_ordering

    def _prep_nac(self, norms: List[float] = None) -> Optional[PredictedSumCalculator]:
        if norms:
            norms = [[i] for i in norms]
            return PredictedSumCalculator(np.array(norms))

    def run(self) -> BemaChangepointResultContainers:
        results = []
        for model in self._models:
            estimator, loads = config.get_changepoint_model_pair(model)
            try:
                self._fit(estimator)
            except Exception as err:
                logger.exception(err)
                e = bema_changepoint_exception_wrapper(
                    err, "A calculation error occurred during modeling.", model=model
                )
                raise e from err

            try:
                result = BemaChangepointResult().create(
                    estimator, loads, self._scorer, self._nac_calc
                )  # XXX TODO error handling -- math could fail
            except Exception as err:
                logger.exception(err)
                e = bema_changepoint_exception_wrapper(
                    err,
                    "A calculation error occurred during post-model calculations.",
                    model=model,
                )
                raise e from err

            results.append(BemaChangepointResultContainer(estimator, result))

        if self._estimator_filter:
            results = self._estimator_filter.filtered(
                results
            )  ## NOTE @tin we must assure this won't error

        return results


def _run_single_batch(
    oat: List[float],
    usage: List[float],
    models: List[str],
    r2_threshold: float,
    cvrmse_threshold: float,
    norms: List[float],
    model_filter: FilterConfig,
) -> BemaChangepointResultContainers:
    if model_filter:
        filt = ChangepointEstimatorFilter(
            which=model_filter.which, how=model_filter.how, extras=model_filter.extras
        )
    else:
        filt = None

    modeler = BemaChangepointModeler(
        oat=oat,
        usage=usage,
        models=models,
        r2_threshold=r2_threshold,
        cvrmse_threshold=cvrmse_threshold,
        norms=norms,
        estimator_filter=filt,
    )

    return modeler.run()


def _format_norms(norms: List[float]) -> nptypes.AnyByAnyNDArray:
    norms = np.array(norms)
    if norms.ndim == 1:  # assure 1d is reshaped according skl spec
        return norms.reshape(-1, 1)
    return np.atleast_2d(norms)


def run_baseline(
    req: BaselineChangepointModelRequest,
) -> EnergyChangepointModelResponse:
    results = _run_single_batch(
        oat=req.usage.oat,
        usage=req.usage.usage,
        models=req.model_config.models,
        r2_threshold=req.model_config.r2_threshold,
        cvrmse_threshold=req.model_config.cvrmse_threshold,
        norms=req.norms,
        model_filter=req.model_config.model_filter,
    )
    return EnergyChangepointModelResponse(results=[r.result for r in results])


def run_optionc(req: SavingsRequest) -> SavingsResponse:
    pre_req = req.pre
    post_req = req.post

    adjcalc = AshraeAdjustedSavingsCalculator(
        confidence_interval=req.confidence_interval, scalar=req.scalar
    )
    print(req.norms)
    normcalc = None
    if req.norms:
        print("heyy")
        X_norms = _format_norms(req.norms)
        normcalc = AshraeNormalizedSavingsCalculator(
            X_norms=X_norms,  # XXX <-- @tin this is what it should look like after https://github.com/cunybpl/changepointmodel/issues/36
            confidence_interval=req.confidence_interval,
            scalar=req.scalar,
        )

    try:
        pre_results = _run_single_batch(
            oat=pre_req.usage.oat,
            usage=pre_req.usage.usage,
            models=pre_req.model_config.models,
            r2_threshold=pre_req.model_config.r2_threshold,
            cvrmse_threshold=pre_req.model_config.cvrmse_threshold,
            norms=req.norms,
            model_filter=pre_req.model_config.model_filter,
        )
    except BemaChangepointException as err:
        e = BemaChangepointException(
            info={"batch": "pre", **err.info}, message=err.message
        )
        raise e from err

    try:
        post_results = _run_single_batch(
            oat=post_req.usage.oat,
            usage=post_req.usage.usage,
            models=post_req.model_config.models,
            r2_threshold=post_req.model_config.r2_threshold,
            cvrmse_threshold=post_req.model_config.cvrmse_threshold,
            norms=req.norms,
            model_filter=post_req.model_config.model_filter,
        )
    except BemaChangepointException as err:
        e = BemaChangepointException(
            info={"batch": "post", **err.info}, message=err.message
        )
        raise e from err

    # NOTE if we don't have pre or post from modeling via a filter we will
    # essentially just skip this set of loops. Which means that we don't need to handle cases in BemaSavingsResult.create
    out = []
    for pre in pre_results:
        for post in post_results:
            try:
                result = BemaSavingsResult().create(pre, post, adjcalc, normcalc)
            except Exception as err:
                logger.exception(err)
                e = bema_changepoint_exception_wrapper(
                    err,
                    "An error occurred while calculating savings.",
                    pre_model=pre.result.json(),
                    post_model=post.result.json(),
                )
                raise e from err
            out.append(result)
    return SavingsResponse(results=out)
