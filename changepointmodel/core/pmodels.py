"""Changepoint parameter model definitions
"""

import abc
import dataclasses
from typing import List, Tuple, Union, TypeVar, Generic
from .nptypes import NByOneNDArray, OneDimNDArray
import numpy as np

from typing_extensions import Protocol

Bound = Tuple[Tuple[float, ...], Tuple[float, ...]]


# defines two callables
class ModelCallable(Protocol):
    def __call__(
        self, X: NByOneNDArray[np.float64], *args: float
    ) -> OneDimNDArray[np.float64]:
        ...


class BoundCallable(Protocol):
    def __call__(
        self, X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
    ) -> Bound:
        ...


@dataclasses.dataclass
class EnergyParameterModelCoefficients(object):
    yint: float
    slopes: List[float]
    changepoints: List[float]


class ICoefficientParser(object):
    @abc.abstractmethod
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        ...


class YinterceptMixin(object):
    def yint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.yint


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


class ISingleSlopeYinterceptModel(ISingleSlopeModel, YinterceptMixin):
    ...


class ISingleSlopeSingleChangepointModel(
    ISingleSlopeModel, ISingleChangepointModel, YinterceptMixin
):
    ...


class IDualSlopeSingleChangepointModel(
    IDualSlopeModel, ISingleChangepointModel, YinterceptMixin
):
    ...


class IDualSlopeDualChangepointModel(
    IDualSlopeModel, IDualChangepointModel, YinterceptMixin
):
    ...


class AbstractEnergyParameterModel(abc.ABC):
    ...  # essentially a namespace


class TwoParameterModel(ISingleSlopeYinterceptModel):
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]


class ThreeParameterModel(ISingleSlopeSingleChangepointModel):
    def slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]


class FourParameterModel(IDualSlopeSingleChangepointModel):
    def left_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def right_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]

    def changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]


class FiveParameterModel(IDualSlopeDualChangepointModel):
    def left_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def right_slope(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]

    def left_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def right_changepoint(self, coeffs: EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[1]


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


# EnergyParameterModel = Union[
#     TwoParameterModel, ThreeParameterModel, FourParameterModel, FiveParameterModel
# ]

EnergyParameterModelT = TypeVar(
    "EnergyParameterModelT",
    TwoParameterModel,
    ThreeParameterModel,
    FourParameterModel,
    FiveParameterModel,
)


class ModelFunction(object):
    def __init__(
        self, name: str, f: ModelCallable, bounds: Union[BoundCallable, Bound]
    ):
        self._name = name
        self._f = f
        self._bounds = bounds

    @property
    def name(self) -> str:
        return self._name

    @property
    def f(self) -> ModelCallable:
        return self._f

    @property
    def bounds(self) -> Union[BoundCallable, Bound]:
        return self._bounds


class ParameterModelFunction(ModelFunction, Generic[EnergyParameterModelT]):
    def __init__(
        self,
        name: str,
        f: ModelCallable,
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

        super().__init__(name, f, bounds)
        self._parameter_model: EnergyParameterModelT = parameter_model
        self._coefficients_parser = coefficients_parser

    @property
    def parameter_model(self) -> EnergyParameterModelT:
        return self._parameter_model

    def parse_coeffs(
        self, coeffs: Tuple[float, ...]
    ) -> EnergyParameterModelCoefficients:
        return self._coefficients_parser.parse(coeffs)
