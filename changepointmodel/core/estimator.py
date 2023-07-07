from typing import Generator, List, Optional, Tuple, Callable, Any, Union, Dict, Generic
import numpy as np

from .nptypes import NByOneNDArray, OneDimNDArray, AnyByAnyNDArray, Ordering
from .pmodels import (
    ParameterModelFunction,
    ParamaterModelCallableT,
    EnergyParameterModelT,
)
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
        bounds: Union[Bounds, OpenBoundCallable, None] = None,
        method: str = "trf",
        jac: Union[
            str, Callable[[npt.NDArray[np.float64], Any], npt.NDArray[np.float64]], None
        ] = None,
        lsq_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_func = model_func
        self.p0 = p0
        self.bounds = bounds
        self.method = method
        self.jac = jac
        self.lsq_kwargs = lsq_kwargs if lsq_kwargs else {}

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
        X, y = check_X_y(X, y)  # type: ignore

        if callable(self.bounds):  # we allow bounds to be a callable
            bounds = self.bounds(X)
        else:
            bounds = self.bounds  # type: ignore

        self.X_ = X
        self.y_ = y

        popt, pcov = optimize.curve_fit(  # type: ignore
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

        self.popt_ = popt  # type: ignore
        self.pcov_ = pcov  # type: ignore
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
        X = check_array(X)  # type: ignore

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

    @classmethod
    def sort_X_y(
        cls, X: NByOneNDArray[np.float64], y: OneDimNDArray[np.float64]
    ) -> Tuple[AnyByAnyNDArray[np.float64], OneDimNDArray[np.float64]]:
        """Helper to sort y in terms of X together for changepoint modeling.

        Args:
            X (NByOneNDArray[np.float64]): The X array
            y (OneDimNDArray[np.float64]): The y array

        Returns:
            Tuple[AnyByAnyNDArray[np.float64], OneDimNDArray[np.float64]]: The reordered X and y
        """
        X, y, _ = argsort_1d_idx(X, y)  # type: ignore
        return X, y

    @classmethod
    def sort_X_y_idx(
        cls, X: NByOneNDArray[np.float64], y: OneDimNDArray[np.float64]
    ) -> Tuple[AnyByAnyNDArray[np.float64], OneDimNDArray[np.float64], Ordering]:
        """A helper to sort y in terms of X together for changepoint modeling. Also returns the
        ordering index that can be assigned to original_ordering field in the estimator or held for later use.

        Args:
            X (NByOneNDArray[np.float64]): _description_
            y (OneDimNDArray[np.float64]): _description_

        Returns:
            Tuple[AnyByAnyNDArray[np.float64], OneDimNDArray[np.float64], Ordering]: _description_
        """
        X, y, order = argsort_1d_idx(X, y)  # type: ignore
        return X, y, order

    @classmethod
    def fit_many(
        cls,
        models: List[
            ParameterModelFunction[ParamaterModelCallableT, EnergyParameterModelT]
        ],
        X: NByOneNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        sigma: Optional[OneDimNDArray[np.float64]] = None,
        absolute_sigma: bool = False,
        sort: bool = True,
        fail_silently: bool = True,
        **estimator_kwargs: Dict[str, Any],
    ) -> Generator[
        Tuple[
            str,
            Optional[
                "EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT]"
            ],
        ],
        None,
        None,
    ]:
        if sort:
            X, y = cls.sort_X_y(X, y)  # type: ignore

        for m in models:
            est = cls(m)
            try:
                yield est.name, est.fit(X, y, sigma, absolute_sigma, **estimator_kwargs)
            except Exception:
                if fail_silently:
                    yield m.name, None
                else:
                    raise

    @classmethod
    def fit_one(
        cls,
        model: ParameterModelFunction[ParamaterModelCallableT, EnergyParameterModelT],
        X: NByOneNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        sigma: Optional[OneDimNDArray[np.float64]] = None,
        absolute_sigma: bool = False,
        sort: bool = True,
        fail_silently: bool = True,
        **estimator_kwargs: Dict[str, Any],
    ) -> Tuple[
        str,
        Optional[
            "EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT]"
        ],
    ]:
        """Fits a single model. Will sort data if needed and optionally fail silently.

        Args:
            model (AbstractEnergyParameterModel): [description]
            data (CurvefitEstimatorData): [description]
            sort (bool, optional): [description]. Defaults to True.

        Returns:
            Optional[AbstractEnergyParameterModel]: [description]
        """
        if sort:
            X, y = cls.sort_X_y(X, y)  # type: ignore

        est = cls(model)
        try:
            return est.name, est.fit(X, y, sigma, absolute_sigma, **estimator_kwargs)
        except (
            Exception
        ):  # XXX  same as above... Note I can't really combine these methods or we'd be sorting the multiple times for no reason
            if fail_silently:
                return model.name, None
            raise

    def fit(
        self,
        X: NByOneNDArray[np.float64],
        y: OneDimNDArray[np.float64],
        sigma: Optional[OneDimNDArray[np.float64]] = None,
        absolute_sigma: bool = False,
    ) -> "EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT]":
        """This is wrapper around CurvefitEstimator.fit and allows interoperability with sklearn
        NOTE: THIS METHOD DOES NOT SORT THE DATA! Input data should be sorted appropriately beforehand. You can
        use changepointmodel.core.utils.argsort_1d for standard changepoint model data.

        Args:
            data (CurvefitEstimatorInputData): [description]
            reshape (Optional[Tuple[int]], optional): [description]. Defaults to None.
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
        if self.model is None:
            raise ValueError("Cannot access name of model that is not set.")
        return self.model.name

    @property
    @check_not_fitted
    def X(self) -> NByOneNDArray[np.float64]:
        return self.X_

    @property
    @check_not_fitted
    def y(self) -> Optional[OneDimNDArray[np.float64]]:
        return self.y_

    @property
    @check_not_fitted
    def coeffs(self) -> Tuple[float, ...]:  # tuple
        return self.estimator_.popt_  # type: ignore

    @property
    @check_not_fitted
    def cov(self) -> Tuple[float, ...]:
        return self.estimator_.pcov_  # type: ignore

    @property
    @check_not_fitted
    def pred_y(self) -> OneDimNDArray[np.float64]:
        return self.pred_y_

    @property
    @check_not_fitted
    def sigma(self) -> Optional[OneDimNDArray[np.float64]]:
        return self.sigma_

    @property
    @check_not_fitted
    def absolute_sigma(self) -> Optional[bool]:
        return self.absolute_sigma_

    @check_not_fitted
    def n_params(self) -> int:
        return len(self.coeffs)

    @check_not_fitted
    def total_pred_y(self) -> float:
        return np.sum(self.pred_y_)  # type: ignore

    @check_not_fitted
    def total_y(self) -> float:
        return np.sum(self.estimator_.y_)  # type: ignore

    @check_not_fitted
    def len_y(self) -> int:
        return len(self.estimator_.y_)  # type: ignore

    @property
    def original_ordering(self) -> Optional[Ordering]:
        return self._original_ordering

    @original_ordering.setter
    def original_ordering(self, o: Ordering) -> None:
        self._original_ordering = o

    # --- added below methods for 3.1
    # TODO need types
    @check_not_fitted
    def r2(self):
        assert self.model is not None
        return self.model.r2(self.y, self.pred_y)

    @check_not_fitted
    def rmse(self):
        assert self.model is not None
        return self.model.rmse(self.y, self.pred_y)

    @check_not_fitted
    def cvrmse(self):
        assert self.model is not None
        return self.model.cvrmse(self.y, self.pred_y)

    @check_not_fitted
    def dpop(self):
        assert self.model is not None
        return self.model.dpop(self.X, self.coeffs)

    @check_not_fitted
    def tstat(self):
        assert self.model is not None
        return self.model.tstat(self.X, self.y, self.pred_y, self.coeffs)

    @check_not_fitted
    def shape(self):
        assert self.model is not None
        return self.model.shape(self.coeffs)

    @check_not_fitted
    def load(self):
        assert self.model is not None
        return self.model.load(self.X, self.pred_y, self.coeffs)
