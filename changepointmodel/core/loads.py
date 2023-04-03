""" Derive the loads from the estimator
"""
import abc
import numpy as np
from .nptypes import OneDimNDArray
from . import utils
from typing import Optional, Tuple, Generic
from .estimator import EnergyChangepointEstimator

from .calc import loads as _loads

from .pmodels import (
    ParamaterModelCallableT,
    EnergyParameterModelT,
    EnergyParameterModelCoefficients,
    ParameterModelFunction,
    TwoParameterModel,
    ThreeParameterModel,
    FourParameterModel,
    FiveParameterModel,
)
from dataclasses import dataclass


@dataclass
class Load(object):
    base: float
    heating: float
    cooling: float


# handles linear and changepoint based loads


class ISensitivityLoad(abc.ABC):
    @abc.abstractmethod
    def calc(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        slope: float,
        yint: float,
        changepoint: Optional[float] = None,
    ) -> float:
        ...

    def __call__(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        slope: float,
        yint: float,
        changepoint: Optional[float] = None,
    ) -> float:
        return self.calc(X, pred_y, slope, yint, changepoint)


class IBaseload(abc.ABC):
    @abc.abstractmethod
    def calc(self, total_consumption: float, *loads: float) -> float:
        ...

    def __call__(self, total_consumption: float, *loads: float) -> float:
        return self.calc(total_consumption, *loads)


class HeatingLoad(ISensitivityLoad):
    def calc(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        slope: float,
        yint: float,
        changepoint: Optional[float] = None,
    ) -> float:
        """Configures and calculates a heating load.

        Args:
            X (OneDimNDArray): The X array input as 1d.
            pred_y (OneDimNDArray): The predicted y values.
            slope (float): The heating slope.
            yint (float): The yintercept.
            changepoint (Optional[float], optional): The heating changepoint.. Defaults to None.

        Returns:
            float: The heating load.
        """
        if slope > 0:  # pos slope no heat
            return 0

        if (
            changepoint is None
        ):  # no changepoint then set to inf  (handles linear model loads)
            changepoint = np.inf

        return _loads.heatload(X, pred_y, yint, changepoint)


class CoolingLoad(ISensitivityLoad):
    def calc(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        slope: float,
        yint: float,
        changepoint: Optional[float] = None,
    ) -> float:
        """Configures and calculates the cooling load for a given model.

        Args:
            X (OneDimNDArray): The X array
            pred_y (OneDimNDArray): The model's predicted y values.
            slope (float): The cooling slope.
            yint (float): The yintercept.
            changepoint (Optional[float], optional): The cooling changepoint. Defaults to None.

        Returns:
            float: [description]
        """
        if slope < 0:  # neg slope no cool
            return 0

        if changepoint is None:  # no cp then set to -inf (handles linear model loads)
            changepoint = -np.inf

        return _loads.coolingload(X, pred_y, yint, changepoint)


class Baseload(IBaseload):
    def calc(self, total_consumption: float, *loads: float) -> float:
        """Calculates the baseload given the total consumption and any number of other loads.

        Args:
            total_consumption (float): The total consumption.

        Returns:
            float: The baseload.
        """
        return _loads.baseload(
            total_consumption, *loads
        )  # This just subtracts all loads from this total y_pred ... subject to change @tynabot


class AbstractLoadHandler(abc.ABC, Generic[EnergyParameterModelT]):
    def __init__(
        self,
        model: EnergyParameterModelT,
        cooling: ISensitivityLoad,
        heating: ISensitivityLoad,
        base: Optional[IBaseload] = None,
    ):
        self._model: EnergyParameterModelT = model
        self._cooling = cooling
        self._heating = heating
        self._base = base if base is not None else Baseload()

    @property
    def model(self) -> EnergyParameterModelT:
        return self._model  # for introspection later.

    def _initial(
        self,
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Tuple[float, np.float64]:
        yint = self._model.yint(coeffs)  # all models need this so placing here...
        total_pred_y = np.sum(pred_y)
        return yint, total_pred_y

    @abc.abstractmethod
    def run(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        ...


class TwoParameterLoadHandler(AbstractLoadHandler[TwoParameterModel]):
    def run(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        """Calculate a two parameter load.

        Args:
            X (OneDimNDArray): [description]
            pred_y (OneDimNDArray): [description]
            coeffs (EnergyParameterModelCoefficients): [description]

        Returns:
            Load: [description]
        """
        yint, total_pred_y = self._initial(pred_y, coeffs)

        slope = self._model.slope(coeffs)
        heating = self._heating(X, pred_y, slope, yint)
        cooling = self._cooling(X, pred_y, slope, yint)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base=base, heating=heating, cooling=cooling)


class ThreeParameterLoadHandler(AbstractLoadHandler[ThreeParameterModel]):
    def run(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        """Calculate a three parameter (heating or cooling) load.

        Args:
            X (OneDimNDArray): The X array of the model.
            pred_y (OneDimNDArray): The predicted y array of the
            coeffs (EnergyParameterModelCoefficients): [description]

        Returns:
            Load: A load calculation.
        """

        yint, total_pred_y = self._initial(pred_y, coeffs)

        slope = self._model.slope(coeffs)
        cp = self._model.changepoint(coeffs)

        heating = self._heating(X, pred_y, slope, yint, cp)
        cooling = self._cooling(X, pred_y, slope, yint, cp)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base, heating, cooling)


class FourParameterLoadHandler(AbstractLoadHandler[FourParameterModel]):
    def run(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        """Calculate a FourParameterLoad.

        Args:
            X (OneDimNDArray): [description]
            pred_y (OneDimNDArray): [description]
            coeffs (EnergyParameterModelCoefficients): [description]

        Returns:
            Load: A Load calculation.
        """

        yint, total_pred_y = self._initial(pred_y, coeffs)

        ls = self._model.left_slope(coeffs)
        rs = self._model.right_slope(coeffs)
        cp = self._model.changepoint(coeffs)

        heating = self._heating(X, pred_y, ls, yint, cp)
        cooling = self._cooling(X, pred_y, rs, yint, cp)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base, heating, cooling)


class FiveParameterLoadHandler(AbstractLoadHandler[FiveParameterModel]):
    def run(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: EnergyParameterModelCoefficients,
    ) -> Load:
        """Calculate a FiveParameterLoad

        Args:
            X (NByOneNDArray): [description]
            pred_y (OneDimNDArray): [description]
            coeffs (EnergyParameterModelCoefficients): [description]

        Returns:
            Load: [description]
        """

        yint, total_pred_y = self._initial(pred_y, coeffs)

        yint = self._model.yint(coeffs)
        ls = self._model.left_slope(coeffs)
        rs = self._model.right_slope(coeffs)
        lcp = self._model.left_changepoint(coeffs)
        rcp = self._model.right_changepoint(coeffs)

        cooling = self._cooling(X, pred_y, rs, yint, rcp)
        heating = self._heating(X, pred_y, ls, yint, lcp)
        base = self._base(float(total_pred_y), cooling, heating)

        return Load(base, heating, cooling)


class EnergyChangepointLoadsAggregator(Generic[EnergyParameterModelT]):
    def __init__(self, handler: AbstractLoadHandler[EnergyParameterModelT]):
        """A high level wrapper around a load handler that works with an
        estimator to aggregate the load.

        Args:
            handler (AbstractLoadHandler): An instance of a load handler.
        """
        self._handler: AbstractLoadHandler[EnergyParameterModelT] = handler

    def aggregate(
        self,
        estimator: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> Load:
        """Performs some type checks between the handler and the estimator and then
        performs the load calculations using the configured handler.

        Args:
            estimator (EnergyChangepointEstimator): A fitted EnergyChangepointEstimator instance.

        Raises:
            TypeError: raised if the estimator's model is not of type ParameterModelFunction.
            TypeError: raised if the estimator's model.pmodel is not of the same type as that of the handler.

        Returns:
            Load: The calculated load
        """

        # check that estimator model type matches the handler's model interface or else we'll have an issue
        # coeff parsing needs to match...
        if not isinstance(
            estimator.model, ParameterModelFunction
        ):  # XXX is there a way around this introspection?
            raise TypeError("estimator.model must be of type ParameterModelFunction")

        if not isinstance(
            estimator.model.parameter_model, self._handler.model.__class__
        ):
            raise TypeError(
                f"estimator parameter_model must be of type {self._handler.model}"
            )

        coeffs = utils.parse_coeffs(estimator.model, estimator.coeffs)
        X = estimator.X.squeeze()  # have to make this 1d
        pred_y = estimator.pred_y
        return self._handler.run(X, pred_y, coeffs)
