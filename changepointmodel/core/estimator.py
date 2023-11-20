from typing import List, Optional, Tuple, Callable, Any, Union, Dict, Generic
import numpy as np

from .nptypes import (
    NByOneNDArray,
    OneDimNDArray,
    AnyByAnyNDArray,
    Ordering,
    SklScoreReturnType,
)
from .pmodels import (
    ParameterModelFunction,
    ParamaterModelCallableT,
    EnergyParameterModelT,
    Load,
)

from .calc import dpop, tstat

from .utils import argsort_1d_idx

import numpy as np
import numpy.typing as npt

from scipy import optimize  # type: ignore

from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError  # type: ignore

from .calc.bounds import BoundTuple, OpenBoundCallable

Bounds = Union[BoundTuple, OpenBoundCallable]


def check_not_fitted(method: Callable[..., Any]) -> Callable[..., Any]:
    """Helper decorator to raise a not fitted error if a property of an Estimator
    is attempted to be accessed before fit.

    Args:
        method (Callable[..., Any]): _description_

    Returns:
        Callable[..., Any]: _description_
    """

    def inner(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        try:
            return method(*args, **kwargs)
        except AttributeError as err:
            raise NotFittedError(
                "This Estimator is not fitted yet. Call `fit` with X and y"
            ) from err

    return inner


class CurvefitEstimator(BaseEstimator, RegressorMixin):  # type: ignore
    def __init__(
        self,
        model_func: Optional[Callable[..., Any]] = None,
        p0: Optional[List[float]] = None,
        bounds: Union[Bounds, OpenBoundCallable, Tuple[float, float], None] = (
            -np.inf,
            np.inf,
        ),
        method: str = "trf",
        jac: Union[
            str, Callable[[npt.NDArray[np.float64], Any], npt.NDArray[np.float64]], None
        ] = None,
        lsq_kwargs: Optional[Dict[Any, Any]] = {},
    ) -> None:
        self.model_func = model_func
        self.p0 = p0
        self.bounds = bounds
        self.method = method
        self.jac = jac
        self.lsq_kwargs = lsq_kwargs  # if lsq_kwargs else {}

    def fit(
        self,
        X: npt.NDArray[np.float64],
        y: Optional[npt.NDArray[np.float64]] = None,
        sigma: Optional[npt.NDArray[np.float64]] = None,
        absolute_sigma: bool = False,
    ) -> "CurvefitEstimator":
        """Fit X features to target y.

        Refer to scipy.optimize.curve_fit docs for details on sigma values.

        Args:
            X (np.array): The feature matrix we are using to fit.
            y (np.array): The target array.
            sigma (Optional[np.array], optional): Determines uncertainty in the ydata. Defaults to None.
            absolute_sigma (bool, optional): Uses sigma in an absolute sense and reflects this in the pcov. Defaults to True.
            squeeze_1d: (bool, optional): Squeeze X into a 1 dimensional array for curve fitting. This is useful if you are fitting
                a function with an X array and do not want to squeeze before it enters curve_fit. Defaults to True.

        Returns:
            GeneralizedCurveFitEstimator: self
        """
        # NOTE the user defined function should handle the neccesary array manipulation (squeeze, reshape etc.)
        # pass the sklearn estimator dimensionality check
        X, y = check_X_y(X, y)

        if callable(self.bounds):  # we allow bounds to be a callable
            bounds = self.bounds(X)
        else:
            bounds = self.bounds  # type: ignore

        self.X_ = X
        self.y_ = y

        popt, pcov = optimize.curve_fit(
            f=self.model_func,
            xdata=X,
            ydata=y,
            p0=self.p0,
            method=self.method,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            bounds=bounds,
            jac=self.jac,
            **self.lsq_kwargs,
        )

        self.popt_ = popt
        self.pcov_ = pcov
        self.name_ = self.model_func.__name__  # type: ignore

        return self

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Predict the target y values given X features using the best fit
        model (model_func) and best fit model parameters (popt)

        Args:
            X (np.array): The X matrix

        Returns:
            np.array: The predicted y values
        """
        check_is_fitted(self, ["popt_", "pcov_", "name_"])
        X = check_array(X)

        return self.model_func(X, *self.popt_)  # type: ignore


class EnergyChangepointEstimator(BaseEstimator, RegressorMixin, Generic[ParamaterModelCallableT, EnergyParameterModelT]):  # type: ignore
    """A container object for a changepoint model. After a model is fit you can access scores and
    load calculations via propeties.
    """

    def __init__(
        self,
        model: Optional[
            ParameterModelFunction[ParamaterModelCallableT, EnergyParameterModelT]
        ] = None,
    ):
        self.model: Optional[
            ParameterModelFunction[ParamaterModelCallableT, EnergyParameterModelT]
        ] = model

        self._original_ordering: Optional[Ordering] = None

    def fit(
        self,
        X: NByOneNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        sigma: Optional[OneDimNDArray[np.float64]] = None,
        absolute_sigma: bool = False,
    ) -> "EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT]":
        """

        This is wrapper around CurvefitEstimator.fit and allows interoperability with sklearn

        NOTE: THIS METHOD DOES NOT SORT THE DATA! Input data should be sorted appropriately beforehand. You can
        use changepointmodel.core.utils.argsort_1d for standard changepoint model data.

        _From 3.1_ use schemas.CurvefitEstimatorDataModel.sorted_X_y to guarantee correct shapes and sorts.

        This method is interoperable with scikit learns API and returns a fit version of itself.

        Args:
            X (NByOneNDArray[np.float64]): The X array. Should be N X 1 dimensions.
            y (OneDimNDArray[np.float64]): The y array. Should be 1 dimension.
            sigma (Optional[OneDimNDArray[np.float64]], optional): An array of weights. Defaults to None.
            absolute_sigma (bool, optional): Whether to use absolute_sigma in fit. Defaults to False.

        Returns:
            EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT]: A fit model.
        """

        self.estimator_ = CurvefitEstimator(
            model_func=self.model.f, bounds=self.model.bounds  # type: ignore
        )
        self.pred_y_ = self.estimator_.fit(X, y, sigma, absolute_sigma).predict(X)

        self.X_, self.y_ = (
            self.estimator_.X_,
            self.estimator_.y_,
        )

        self.sigma_ = sigma
        self.absolute_sigma_ = absolute_sigma

        return self

    def predict(self, X: NByOneNDArray[np.float64]) -> OneDimNDArray[np.float64]:
        """Proxy a call to estimator.predict in order to use the model to generate a changepoint model with
        different X vals on the fit estimator. This also allows interoperability with sklearn.

        Args:
            data (CurvefitEstimatorRegressorData): [description]

        Returns:
            np.array : predicted y values
        """
        check_is_fitted(self)
        return self.estimator_.predict(X)

    def adjust(
        self,
        other: "EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT]",
    ) -> OneDimNDArray[np.float64]:
        """A convenience method that predicts using the X values of another EnergyChangepointEstimator.
        In option-c methodology this other would be the post retrofit model, making this calling instance the pre model.

        Args:
            other (EnergyChangepointEstimator): [description]

        Returns:
            OneDimNDArray: [description]
        """
        check_is_fitted(self)
        return self.predict(other.X)

    @property
    def name(self) -> str:
        """The name given to this model. Our default energy models are 2P, 3PC, 3PH, 4P and 5P.

        Raises:
            ValueError: _description_

        Returns:
            str: _description_
        """
        if self.model is None:
            raise ValueError("Cannot access name of model that is not set.")
        return self.model.name

    @property
    @check_not_fitted
    def X(self) -> NByOneNDArray[np.float64]:
        """Original X array usef for fit.

        Returns:
            NByOneNDArray[np.float64]: _description_
        """
        return self.X_

    @property
    @check_not_fitted
    def y(self) -> Optional[OneDimNDArray[np.float64]]:
        """Original y array used for fit.

        Returns:
            Optional[OneDimNDArray[np.float64]]: _description_
        """
        return self.y_

    @property
    @check_not_fitted
    def coeffs(self) -> Tuple[float, ...]:  # tuple
        """The fit model coefficients. This is returned by
        scipy.optimize.curve_fit popt variable

        Returns:
            Tuple[float, ...]: _description_
        """
        return self.estimator_.popt_  # type: ignore

    @property
    @check_not_fitted
    def cov(self) -> Tuple[float, ...]:
        """The covariance of the fit. Returned by
        scipy.optimize.curve_fit pcov variable.

        Returns:
            Tuple[float, ...]: _description_
        """
        return self.estimator_.pcov_  # type: ignore

    @property
    @check_not_fitted
    def pred_y(self) -> OneDimNDArray[np.float64]:
        """The predicted y values. We automatically generate these on fit
        to make this accessible.

        Returns:
            OneDimNDArray[np.float64]: _description_
        """
        return self.pred_y_

    @property
    @check_not_fitted
    def sigma(self) -> Optional[OneDimNDArray[np.float64]]:
        """Reference to the sigma array (weights) if used in model fitting.

        Returns:
            Optional[OneDimNDArray[np.float64]]: _description_
        """
        return self.sigma_

    @property
    @check_not_fitted
    def absolute_sigma(self) -> Optional[bool]:
        """Reference to the absolute sigma if this was used in model fitting.

        Returns:
            Optional[bool]: _description_
        """
        return self.absolute_sigma_

    @check_not_fitted
    def n_params(self) -> int:
        """The number of model coefficients.

        Returns:
            int: _description_
        """
        return len(self.coeffs)

    @check_not_fitted
    def total_pred_y(self) -> float:
        """The sum of the predicted y value generated
        generated after fit.

        Returns:
            float: _description_
        """
        return np.sum(self.pred_y_)  # type: ignore

    @check_not_fitted
    def total_y(self) -> float:
        """The sum of the y array.

        Returns:
            float: _description_
        """
        return np.sum(self.estimator_.y_)  # type: ignore

    @check_not_fitted
    def len_y(self) -> int:
        """The length of the y array.

        Returns:
            int: _description_
        """
        return len(self.estimator_.y_)  # type: ignore

    @property
    def original_ordering(self) -> Optional[Ordering]:
        """A place to append the original ordering of the X, y
        data for this model if needed. Can be used with `unargsort` method
        found in utils for reversing joined sort.

        Returns:
            Optional[Ordering]: An order index given by utils.argsort_1d_idx
        """
        return self._original_ordering

    @original_ordering.setter
    def original_ordering(self, o: Ordering) -> None:
        self._original_ordering = o

    # --- added below methods for 3.1
    @check_not_fitted
    def r2(self) -> SklScoreReturnType:
        """The r2 of the model. Determines goodness of fit.

        Returns:
            SklScoreReturnType: The score (usually a float)
        """
        assert self.model is not None
        return self.model.r2(self.y, self.pred_y)

    @check_not_fitted
    def adjusted_r2(self) -> SklScoreReturnType:
        """The adjusted_r2 of the model. Determines goodness of fit.

        Returns:
            SklScoreReturnType: The score (usually a float)
        """
        assert self.model is not None
        return self.model.adjusted_r2(self.y, self.pred_y, self.coeffs)

    @check_not_fitted
    def rmse(self) -> SklScoreReturnType:
        """The rmse of the model. The amount error in the model.

        Returns:
            SklScoreReturnType: The score (usually a float)
        """
        assert self.model is not None
        return self.model.rmse(self.y, self.pred_y)

    @check_not_fitted
    def cvrmse(self) -> SklScoreReturnType:
        """The cvrmse of the model. The amount of error in the model.

        Returns:
            SklScoreReturnType: The score (usually a float)
        """
        assert self.model is not None
        return self.model.cvrmse(self.y, self.pred_y)

    @check_not_fitted
    def dpop(self) -> dpop.HeatingCoolingPoints:
        """Returns the number of heating and cooling points for the
        given energy model.

        Returns:
            dpop.HeatingCoolingPoints: The heating and cooling points.
        """
        assert self.model is not None
        return self.model.dpop(self.X.squeeze(), self.coeffs)

    @check_not_fitted
    def tstat(self) -> tstat.HeatingCoolingTStatResult:
        """A tstat that can be used to determine if a model's heating
        and/or cooling slopes are statistcally significant.

        Returns:
            tstat.HeatingCoolingTStatResult: The heating and cooling tstat
        """
        assert self.model is not None
        return self.model.tstat(self.X.squeeze(), self.y, self.pred_y, self.coeffs)

    @check_not_fitted
    def shape(self) -> bool:
        """Assert whether this model fits a proper "shape" of an energy model.
        This is often only needed to determine certain inconsistencies between
        4P and 5P models that curve_fit bounds cannot mathematically ascertain in edge
        case data.

        Returns:
            bool: True if it passes the shape test.
        """
        assert self.model is not None
        return self.model.shape(self.coeffs)

    @check_not_fitted
    def load(self, scalar: Optional[float] = None) -> Load:
        """Calculate the load for this energy model. If you are working with monthly data and
        your original model was fit using per_day values provide scalar with `30.437` to scale
        the result to gross total for the modeling period.

        Args:
            scalar (Optional[float], optional): A scalar value. Set to `30.437` to scale avg_day_per_month to gross month.
                Defaults to None.

        Returns:
            Load: The baseload, heating and cooling load.
        """
        assert self.model is not None

        load = self.model.load(self.X.squeeze(), self.pred_y, self.coeffs)
        if scalar:
            load.base *= scalar
            load.cooling *= scalar
            load.heating *= scalar
        return load

    @check_not_fitted
    def nac(
        self, X: NByOneNDArray[np.float64], scalar: Optional[float] = None
    ) -> float:
        """Normalized Annual consumption calculation. If you are working with monthly data and
        your original model was fit using per_day values provide scalar with `30.437` to scale
        the result to gross total for the modeling period.

        Args:
            X (NByOneNDArray[np.float64]): The X array. Usually normalized temperature data (monthly)
            scalar (Optional[float], optional): A scalar value. Set to `30.437` to scale avg_day_per_month to gross month.
                Defaults to None.

        Returns:
            float: The nac value. The sum of the y_pred given your normalized values multiplied by scalar if provided.
        """
        assert self.model is not None

        if scalar:
            return float(np.sum(self.estimator_.predict(X))) * scalar
        else:
            return float(np.sum(self.estimator_.predict(X)))
