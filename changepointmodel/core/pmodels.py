"""Changepoint parameter model definitions
"""

import abc
import dataclasses
from typing import List, Tuple, Union, TypeVar, Generic, Callable, Optional
from .nptypes import NByOneNDArray, OneDimNDArray
import numpy as np

from .calc import tstat, dpop
from .calc import metrics
from .calc import loads

from typing_extensions import Protocol


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


class YinterceptMixin(object):
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.yint


class ICoefficientParser(object):
    @abc.abstractmethod
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        ...


class TwoParameterCoefficientParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, slope = coeffs
        return EnergyParameterModelCoefficients(yint, [slope], [])


class ThreeParameterCoefficientsParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, slope, changepoint = coeffs
        return EnergyParameterModelCoefficients(yint, [slope], [changepoint])


class FourParameterCoefficientsParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, ls, rs, changepoint = coeffs
        return EnergyParameterModelCoefficients(yint, [ls, rs], [changepoint])


class FiveParameterCoefficientsParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, ls, rs, lcp, rcp = coeffs
        return EnergyParameterModelCoefficients(yint, [ls, rs], [lcp, rcp])


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


# extension interface for Tstat
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


class AllMetricsMixin(
    R2MetricMixin,
    CvRmseMetricMixin,
    RmseMetricMixin,
):
    ...


@dataclasses.dataclass
class Load(object):
    base: float
    heating: float
    cooling: float


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


# NOTE that calls to slope and changepoint can work internally by return EnergyParameterModelCoefficients.
# so coeffs parser would be used internally(model specific calcs) and externally (reporting from Estimator)
class AbstractEnergyParameterModel(
    AllMetricsMixin,
    IShapeTestMixin,
    IDataPopMixin,
    ITStatMixin,
    YinterceptMixin,
    ILoad,
):
    ...


class TwoParameterModel(
    AbstractEnergyParameterModel,
    ISingleSlopeModel,
):
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.twop(X, y, pred_y, self.slope(coeffs))

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.twop(X, self.slope(coeffs))

    def shape(self, coeffs: EnergyParameterModelCoefficients) -> bool:
        # essentially this is a no -op / constant
        return True

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        slope = self.slope(coeffs)

        heating = self._heating(X, pred_y, slope, yint)
        cooling = self._cooling(X, pred_y, slope, yint)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base=base, heating=heating, cooling=cooling)


class ThreeParameterCoolingModel(
    AbstractEnergyParameterModel,
    ISingleSlopeSingleChangepointModel,
):
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.threepc(X, y, pred_y, self.slope(coeffs), self.changepoint(coeffs))

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.threepc(X, self.changepoint(coeffs))

    def shape(self, coeffs: EnergyParameterModelCoefficients) -> bool:
        if self.slope(coeffs) > 0:
            return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        slope = self.slope(coeffs)
        cp = self.changepoint(coeffs)

        heating = self._heating(X, pred_y, slope, yint, cp)
        cooling = self._cooling(X, pred_y, slope, yint, cp)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base, heating, cooling)


class ThreeParameterHeatingModel(
    AbstractEnergyParameterModel,
    ISingleSlopeSingleChangepointModel,
):
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.threeph(X, y, pred_y, self.slope(coeffs), self.changepoint(coeffs))

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.threeph(X, self.changepoint(coeffs))

    def shape(self, coeffs: EnergyParameterModelCoefficients) -> bool:
        if self.slope(coeffs) < 0:
            return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        slope = self.slope(coeffs)
        cp = self.changepoint(coeffs)

        heating = self._heating(X, pred_y, slope, yint, cp)
        cooling = self._cooling(X, pred_y, slope, yint, cp)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base, heating, cooling)


class FourParameterModel(
    AbstractEnergyParameterModel,
    IDualSlopeSingleChangepointModel,
):
    def left_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def right_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]

    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.fourp(
            X,
            y,
            pred_y,
            self.left_slope(coeffs),
            self.right_slope(coeffs),
            self.changepoint(coeffs),
        )

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.fourp(X, self.changepoint(coeffs))

    def shape(self, coeffs: EnergyParameterModelCoefficients) -> bool:
        ls, rs = self.left_slope(coeffs), self.right_slope(coeffs)
        if ls < 0 and rs > 0:  # should be V shape
            if abs(ls) > abs(rs):  # check the magnitude of the slopes
                return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        ls = self.left_slope(coeffs)
        rs = self.right_slope(coeffs)
        cp = self.changepoint(coeffs)

        heating = self._heating(X, pred_y, ls, yint, cp)
        cooling = self._cooling(X, pred_y, rs, yint, cp)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base, heating, cooling)


class FiveParameterModel(
    AbstractEnergyParameterModel,
    IDualSlopeDualChangepointModel,
):
    def left_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def right_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]

    def left_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def right_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[1]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.fivep(
            X,
            y,
            pred_y,
            self.left_slope(coeffs),
            self.right_slope(coeffs),
            self.left_changepoint(coeffs),
            self.right_changepoint(coeffs),
        )

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.fivep(
            X, self.left_changepoint(coeffs), self.right_changepoint(coeffs)
        )

    def shape(self, coeffs: EnergyParameterModelCoefficients) -> bool:
        ls, rs = self.left_slope(coeffs), self.right_slope(coeffs)
        if ls < 0 and rs > 0:  # should be V shape
            if abs(ls) > abs(rs):  # check the magnitude of the slopes
                return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        ls = self.left_slope(coeffs)
        rs = self.right_slope(coeffs)
        lcp = self.left_changepoint(coeffs)
        rcp = self.right_changepoint(coeffs)

        cooling = self._cooling(X, pred_y, rs, yint, rcp)
        heating = self._heating(X, pred_y, ls, yint, lcp)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base, heating, cooling)


EnergyParameterModelT = TypeVar(
    "EnergyParameterModelT", bound=AbstractEnergyParameterModel
)

# a simpler loads interface that we can directly to each parameter model..


class ParameterModelFunction(
    Generic[ParamaterModelCallableT, EnergyParameterModelT],
):
    def __init__(
        self,
        name: str,
        f: ParamaterModelCallableT,
        bounds: Union[BoundCallable, Bound],
        parameter_model: EnergyParameterModelT,
        coefficients_parser: ICoefficientParser,
    ):
        """A Parameter model function for our changepoint modeling is composed
        of a callable "model" function (This is most likely 1d), Bounds, EnergyParameterModel
        and CoefficientsParser. These must be configured at runtime for each available model
        to run in an application in order to get the benefits of our API.

        Args:
            name (str): _description_
            f (ModelCallable): _description_
            bounds (Union[BoundCallable, Bound]): _description_
            parameter_model (EnergyParameterModel): _description_
            coefficients_parser (ICoefficientParser): _description_
        """
        self._name = name
        self._f: ParamaterModelCallableT = f
        self._bounds = bounds
        self._parameter_model: EnergyParameterModelT = parameter_model
        self._coefficients_parser = coefficients_parser

    @property
    def name(self) -> str:
        return self._name

    @property
    def f(self) -> ParamaterModelCallableT:
        return self._f

    @property
    def bounds(self) -> Union[BoundCallable, Bound]:
        return self._bounds

    @property
    def parameter_model(self) -> EnergyParameterModelT:
        return self._parameter_model

    def parse_coeffs(
        self, coeffs: Tuple[float, ...]
    ) -> EnergyParameterModelCoefficients:
        return self._coefficients_parser.parse(coeffs)

    # XXX  v3.1

    def r2(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return self._parameter_model.r2(y, y_pred)

    def rmse(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return self._parameter_model.rmse(y, y_pred)

    def cvrmse(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return self._parameter_model.rmse(y, y_pred)

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: Tuple[float, ...],
    ) -> tstat.HeatingCoolingTStatResult:
        return self._parameter_model.tstat(X, y, pred_y, self.parse_coeffs(coeffs))

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: Tuple[float, ...],
    ) -> dpop.HeatingCoolingPoints:
        return self._parameter_model.dpop(X, self.parse_coeffs(coeffs))

    def shape(self, coeffs: Tuple[float, ...]) -> bool:
        return self._parameter_model.shape(self.parse_coeffs(coeffs))

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: Tuple[float, ...],
    ) -> Load:
        return self._parameter_model.load(X, pred_y, self.parse_coeffs(coeffs))


# TODO factory methods for ParameterModelFunctionExt ... makes concrete from generic class above.


# def twop_model(name='2P') -> ParameterModelFunction[]
