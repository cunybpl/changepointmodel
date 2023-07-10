from . import base
from ..nptypes import OneDimNDArray
import numpy as np

from typing import Union, Tuple, Generic

from ..calc import tstat, dpop


class TwoParameterModel(
    base.AbstractEnergyParameterModel,
    base.ISingleSlopeModel,
):
    def slope(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.twop(X, y, pred_y, self.slope(coeffs))

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.twop(X, self.slope(coeffs))

    def shape(self, coeffs: base.EnergyParameterModelCoefficients) -> bool:
        # essentially this is a no -op / constant
        return True

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> base.Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        slope = self.slope(coeffs)

        heating = self._heating(X, pred_y, slope, yint)
        cooling = self._cooling(X, pred_y, slope, yint)
        base_ = self._base(float(total_pred_y), cooling, heating)

        return base.Load(base=base_, heating=heating, cooling=cooling)


class ThreeParameterCoolingModel(
    base.AbstractEnergyParameterModel,
    base.ISingleSlopeSingleChangepointModel,
):
    def slope(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def changepoint(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.threepc(X, y, pred_y, self.slope(coeffs), self.changepoint(coeffs))

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.threepc(X, self.changepoint(coeffs))

    def shape(self, coeffs: base.EnergyParameterModelCoefficients) -> bool:
        if self.slope(coeffs) > 0:
            return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> base.Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        slope = self.slope(coeffs)
        cp = self.changepoint(coeffs)

        heating = self._heating(X, pred_y, slope, yint, cp)
        cooling = self._cooling(X, pred_y, slope, yint, cp)
        base_ = self._base(float(total_pred_y), cooling, heating)

        return base.Load(base_, heating, cooling)


class ThreeParameterHeatingModel(
    base.AbstractEnergyParameterModel,
    base.ISingleSlopeSingleChangepointModel,
):
    def slope(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def changepoint(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> tstat.HeatingCoolingTStatResult:
        return tstat.threeph(X, y, pred_y, self.slope(coeffs), self.changepoint(coeffs))

    def dpop(
        self,
        X: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.threeph(X, self.changepoint(coeffs))

    def shape(self, coeffs: base.EnergyParameterModelCoefficients) -> bool:
        if self.slope(coeffs) < 0:
            return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> base.Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        slope = self.slope(coeffs)
        cp = self.changepoint(coeffs)

        heating = self._heating(X, pred_y, slope, yint, cp)
        cooling = self._cooling(X, pred_y, slope, yint, cp)
        base_ = self._base(float(total_pred_y), cooling, heating)

        return base.Load(base_, heating, cooling)


class FourParameterModel(
    base.AbstractEnergyParameterModel,
    base.IDualSlopeSingleChangepointModel,
):
    def left_slope(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def right_slope(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]

    def changepoint(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
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
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.fourp(X, self.changepoint(coeffs))

    def shape(self, coeffs: base.EnergyParameterModelCoefficients) -> bool:
        ls, rs = self.left_slope(coeffs), self.right_slope(coeffs)
        if ls < 0 and rs > 0:  # should be V shape
            if abs(ls) > abs(rs):  # check the magnitude of the slopes
                return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> base.Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        ls = self.left_slope(coeffs)
        rs = self.right_slope(coeffs)
        cp = self.changepoint(coeffs)

        heating = self._heating(X, pred_y, ls, yint, cp)
        cooling = self._cooling(X, pred_y, rs, yint, cp)
        base_ = self._base(float(total_pred_y), cooling, heating)

        return base.Load(base_, heating, cooling)


class FiveParameterModel(
    base.AbstractEnergyParameterModel,
    base.IDualSlopeDualChangepointModel,
):
    def left_slope(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[0]

    def right_slope(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.slopes[1]

    def left_changepoint(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[0]

    def right_changepoint(self, coeffs: base.EnergyParameterModelCoefficients) -> float:
        return coeffs.changepoints[1]

    def tstat(
        self,
        X: OneDimNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
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
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> dpop.HeatingCoolingPoints:
        return dpop.fivep(
            X, self.left_changepoint(coeffs), self.right_changepoint(coeffs)
        )

    def shape(self, coeffs: base.EnergyParameterModelCoefficients) -> bool:
        ls, rs = self.left_slope(coeffs), self.right_slope(coeffs)
        if ls < 0 and rs > 0:  # should be V shape
            if abs(ls) > abs(rs):  # check the magnitude of the slopes
                return True
        return False

    def load(
        self,
        X: OneDimNDArray[np.float64],
        pred_y: OneDimNDArray[np.float64],
        coeffs: base.EnergyParameterModelCoefficients,
    ) -> base.Load:
        yint = self.yint(coeffs)
        total_pred_y = np.sum(pred_y)
        ls = self.left_slope(coeffs)
        rs = self.right_slope(coeffs)
        lcp = self.left_changepoint(coeffs)
        rcp = self.right_changepoint(coeffs)

        cooling = self._cooling(X, pred_y, rs, yint, rcp)
        heating = self._heating(X, pred_y, ls, yint, lcp)
        base_ = self._base(float(total_pred_y), cooling, heating)

        return base.Load(base_, heating, cooling)


# a simpler loads interface that we can directly to each parameter model..


class ParameterModelFunction(
    Generic[base.ParamaterModelCallableT, base.EnergyParameterModelT],
):
    def __init__(
        self,
        name: str,
        f: base.ParamaterModelCallableT,
        bounds: Union[base.BoundCallable, base.Bound],
        parameter_model: base.EnergyParameterModelT,
        coefficients_parser: base.ICoefficientParser,
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
        self._f: base.ParamaterModelCallableT = f
        self._bounds = bounds
        self._parameter_model = parameter_model
        self._coefficients_parser = coefficients_parser

    @property
    def name(self) -> str:
        return self._name

    @property
    def f(self) -> base.ParamaterModelCallableT:
        return self._f

    @property
    def bounds(self) -> Union[base.BoundCallable, base.Bound]:
        return self._bounds

    @property
    def parameter_model(self) -> base.EnergyParameterModelT:
        return self._parameter_model

    def parse_coeffs(
        self, coeffs: Tuple[float, ...]
    ) -> base.EnergyParameterModelCoefficients:
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

    def adjusted_r2(
        self,
        y: OneDimNDArray[np.float64],
        y_pred: OneDimNDArray[np.float64],
        coeffs: Tuple[float, ...],
    ) -> Union[float, OneDimNDArray[np.float64]]:
        return self._parameter_model.adjusted_r2(y, y_pred, self.parse_coeffs(coeffs))

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
    ) -> base.Load:
        return self._parameter_model.load(X, pred_y, self.parse_coeffs(coeffs))
