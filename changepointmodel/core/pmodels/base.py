from typing import List, Tuple, Union, Optional
import dataclasses

from typing import List, Tuple, Union, TypeVar, Generic, Callable, Optional, Protocol
from ..nptypes import NByOneNDArray, OneDimNDArray
import numpy as np
import dataclasses

from ..nptypes import OneDimNDArray

import numpy as np

from ..calc import dpop, tstat, metrics, loads

import abc


Bound = Tuple[Tuple[float, ...], Tuple[float, ...]]

TwoParameterCallable = Callable[
    [NByOneNDArray[np.float64], float, float], OneDimNDArray[np.float64]
]
ThreeParameterCallable = Callable[
    [NByOneNDArray[np.float64], float, float, float], OneDimNDArray[np.float64]
]
FourParameterCallable = Callable[
    [NByOneNDArray[np.float64], float, float, float, float], OneDimNDArray[np.float64]
]
FiveParameterCallable = Callable[
    [NByOneNDArray[np.float64], float, float, float, float, float],
    OneDimNDArray[np.float64],
]


ParamaterModelCallableT = TypeVar(
    "ParamaterModelCallableT",
    TwoParameterCallable,
    ThreeParameterCallable,
    FourParameterCallable,
    FiveParameterCallable,
)


class BoundCallable(Protocol):
    def __call__(
        self, X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
    ) -> Bound:
        ...


@dataclasses.dataclass
class EnergyParameterModelCoefficients(object):
    yint: float
    slopes: List[float]
    changepoints: List[float] = dataclasses.field(default_factory=lambda: [])

    def n_params(self) -> int:
        return 1 + len(self.slopes) + len(self.changepoints)


@dataclasses.dataclass
class Load(object):
    base: float
    heating: float
    cooling: float


class YinterceptMixin(object):
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.yint


class ICoefficientParser(object):
    @abc.abstractmethod
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        ...


class ISingleSlopeModel(abc.ABC):
    @abc.abstractmethod
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        ...


class ISingleChangepointModel(abc.ABC):
    @abc.abstractmethod
    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        ...


class IDualSlopeModel(abc.ABC):
    @abc.abstractmethod
    def left_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        ...

    @abc.abstractmethod
    def right_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        ...


class IDualChangepointModel(abc.ABC):
    @abc.abstractmethod
    def left_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        ...

    @abc.abstractmethod
    def right_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        ...


class ISingleSlopeSingleChangepointModel(ISingleSlopeModel, ISingleChangepointModel):
    ...


class IDualSlopeSingleChangepointModel(IDualSlopeModel, ISingleChangepointModel):
    ...


class IDualSlopeDualChangepointModel(IDualSlopeModel, IDualChangepointModel):
    ...


class ITStatMixin(abc.ABC):
    @abc.abstractmethod
    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        ...


# We can use 1 Dpop mixin since there is 1 return type.
class IDataPopMixin(abc.ABC):
    @abc.abstractmethod
    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        ...


# Shape test has... 4 arg implementations ...  1 return type (bool) ...
class IShapeTestMixin(abc.ABC):
    @abc.abstractmethod
    def shape(self, coeffs: EnergyParameterModelCoefficients) -> bool:
        ...


class R2MetricMixin(object):
    def r2(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return metrics.r2_score(y, y_pred)


class RmseMetricMixin(object):
    def rmse(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return metrics.rmse(y, y_pred)


class CvRmseMetricMixin(object):
    def cvrmse(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return metrics.cvrmse(y, y_pred)


class AdjR2MetricMixin(object):
    def adjusted_r2(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return metrics.adjusted_r2_score(y, y_pred, coeffs.n_params())


class ILoad(abc.ABC):
    def _base(self, total_consumption: float, *loads_: float) -> float:
        return loads.baseload(total_consumption, *loads_)

    def _heating(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        slope: float,
        yint: float,
        changepoint: Optional[float] = None,
    ) -> float:
        if slope > 0:  # pos slope no heat
            return 0

        # no changepoint then set to inf  (handles linear model loads)
        if changepoint is None:
            changepoint = np.inf

        return loads.heatload(X, pred_y, yint, changepoint)

    def _cooling(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        slope: float,
        yint: float,
        changepoint: Optional[float] = None,
    ) -> float:
        if slope < 0:  # neg slope no cool
            return 0

        if changepoint is None:  # no cp then set to -inf (handles linear model loads)
            changepoint = -np.inf

        return loads.coolingload(X, pred_y, yint, changepoint)

    @abc.abstractmethod
    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        ...


class AllMetricsMixin(
    R2MetricMixin,
    CvRmseMetricMixin,
    RmseMetricMixin,
    AdjR2MetricMixin,
):
    ...


class AbstractEnergyParameterModel(
    AllMetricsMixin,
    IShapeTestMixin,
    IDataPopMixin,
    ITStatMixin,
    YinterceptMixin,
    ILoad,
):
    ...


EnergyParameterModelT = TypeVar(
    "EnergyParameterModelT", bound=AbstractEnergyParameterModel
)
