from changepointmodel.core.utils import parse_coeffs, unargsort_1d_idx
from .models import (
    EnergyChangepointModelResult,
    AdjustedSavingsResultData,
    SavingsResult,
    NormalizedSavingsResultData,
)
from changepointmodel.core.estimator import EnergyChangepointEstimator
from changepointmodel.core.loads import EnergyChangepointLoadsAggregator
from changepointmodel.core.scoring import Scorer

from changepointmodel.core.predsum import PredictedSumCalculator
from changepointmodel.core.savings import (
    AshraeAdjustedSavingsCalculator,
    AshraeNormalizedSavingsCalculator,
)

from .base import BemaChangepointResultContainer
from typing import Optional

from changepointmodel.core.pmodels import ParamaterModelCallableT, EnergyParameterModelT


class BemaChangepointResult(object):
    def create(
        self,
        estimator: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
        loads: EnergyChangepointLoadsAggregator[EnergyParameterModelT],
        scorer: Scorer,
        nac: Optional[PredictedSumCalculator],
    ) -> EnergyChangepointModelResult:
        # XXX change this access to public #67
        original_ordering = estimator.original_ordering
        assert original_ordering is not None, "original ordering is None"
        assert estimator.model is not None, "estimator model is None"

        data = {
            "name": estimator.name,
            "input_data": {
                "X": unargsort_1d_idx(estimator.X, original_ordering),
                "y": unargsort_1d_idx(estimator.y, original_ordering),
            },
            # XXX some of these are dataclasses and doesn't integrate well with pydantic, .___dict___ for now
            "coeffs": vars(parse_coeffs(estimator.model, estimator.coeffs)),
            "pred_y": unargsort_1d_idx(estimator.pred_y, original_ordering),
            "load": vars(loads.aggregate(estimator)),
            "scores": [vars(i) for i in scorer.check(estimator)],
            "nac": vars(nac.calculate(estimator)) if nac else None,
        }

        return EnergyChangepointModelResult(**data)  # type: ignore


class BemaSavingsResult(object):
    def create(
        self,
        pre: BemaChangepointResultContainer[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
        post: BemaChangepointResultContainer[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
        adjcalc: AshraeAdjustedSavingsCalculator,
        normcalc: Optional[AshraeNormalizedSavingsCalculator] = None,
    ) -> SavingsResult:
        result = adjcalc.save(pre.estimator, post.estimator)

        ordering = post.estimator.original_ordering
        assert ordering is not None, "ordering is None."

        result.adjusted_y = unargsort_1d_idx(result.adjusted_y, ordering)

        adj = AdjustedSavingsResultData(
            result=vars(result), confidence_interval=adjcalc.confidence_interval  # type: ignore
        )

        # XXX this is only calculated if norms were provided... otherwise it returns null
        if normcalc:
            result_ = normcalc.save(pre.estimator, post.estimator)
            norm = NormalizedSavingsResultData(
                confidence_interval=normcalc.confidence_interval, result=vars(result_)  # type: ignore
            )
        else:
            norm = None

        data = {
            "pre": pre.result,
            "post": post.result,
            "adjusted_savings": adj,
            "normalized_savings": norm,
        }
        return SavingsResult(**data)  # type: ignore
