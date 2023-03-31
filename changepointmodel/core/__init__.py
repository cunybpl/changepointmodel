from .estimator import EnergyChangepointEstimator, CurvefitEstimator
from .factories import EnergyModel, EnergyModelFactory
from .loads import (
    EnergyChangepointLoadsAggregator,
    FiveParameterLoadHandler,
    FourParameterLoadHandler,
    ThreeParameterLoadHandler,
    TwoParameterLoadHandler,
    CoolingLoad,
    HeatingLoad,
    Baseload,
)
from .pmodels import (
    TwoParameterModel,
    TwoParameterCoefficientParser,
    ThreeParameterModel,
    ThreeParameterCoefficientsParser,
    FourParameterModel,
    FourParameterCoefficientsParser,
    FiveParameterModel,
    FiveParameterCoefficientsParser,
    ParameterModelFunction,
)
from .predsum import PredictedSumCalculator
from .savings import AshraeAdjustedSavingsCalculator, AshraeNormalizedSavingsCalculator
from .schemas import CurvefitEstimatorDataModel
from .scoring import R2, Rmse, Cvrmse, Scorer, ScoreEval
from .utils import argsort_1d_idx, unargsort_1d_idx, parse_coeffs


_loads = (
    "EnergyChangepointLoadsAggregator",
    "FiveParameterLoadHandler",
    "FourParameterLoadHandler",
    "ThreeParameterLoadHandler",
    "TwoParameterLoadHandler",
    "CoolingLoad",
    "HeatingLoad",
    "Baseload",
)

_pmodels = (
    "TwoParameterModel",
    "TwoParameterCoefficientParser",
    "ThreeParameterModel",
    "ThreeParameterCoefficientsParser",
    "FourParameterModel",
    "FourParameterCoefficientsParser",
    "FiveParameterModel",
    "FiveParameterCoefficientsParser",
    "ParameterModelFunction",
)

_scoring = ("R2", "Rmse", "Cvrmse", "Scorer", "ScoreEval")

__all__ = (
    "EnergyChangepointEstimator",
    "CurvefitEstimator",
    "EnergyModel",
    "EnergyModelFactory",
    "PredictedSumCalculator",
    "AshraeAdjustedSavingsCalculator",
    "AshraeNormalizedSavingsCalculator",
    "CurvefitEstimatorDataModel",
    "argsort_1d_idx",
    "unargsort_1d_idx",
    "parse_coeffs",
    *_loads,
    *_pmodels,
    *_scoring,
)
