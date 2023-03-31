from typing import Generator, List, Optional, Tuple, Callable, Any, Union
import numpy as np

from .nptypes import NByOneNDArray, OneDimNDArray
from .pmodels import ModelFunction
from .utils import argsort_1d_idx

import numpy as np

from scipy import optimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError


def check_not_fitted(method):
    def inner(*args, **kwargs):
        try:
            return method(*args, **kwargs)
        except AttributeError as err:
            raise NotFittedError(
                f"This Estimator is not fitted yet. Call `fit` with X and y"
            ) from err

    return inner


class CurvefitEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        model_func: Callable[[], np.array] = None,
        p0: Optional[List[float]] = None,
        bounds: Union[
            Tuple[np.dtype, np.dtype],
            List[Tuple[np.dtype, np.dtype]],
            Callable[[], List[Tuple[np.dtype, np.dtype]]],
        ] = (-np.inf, np.inf),
        method: str = "trf",
        jac: Union[str, Callable[[np.array, Any], np.array], None] = None,
        lsq_kwargs: dict = None,
    ) -> None:
        """Wraps the scipy.optimize.curve_fit function for non-linear least squares. The curve_fit function is itself a wrapper around
        scipy.optimize.leastsq and/or scipy.optimize.least_squares that aims to simplfy some of the calling mechanisms. An entrypoint
        to kwargs for these lower level method is provided by the lsq_kwargs dictionary.

        On success, the curve_fit function will return a tuple of the optimized parameters to the function (popt) as well as the estimated
        covariance of these parameters. These values are used in the predict method and can be accessed after the model has been fit
            as ``model.popt_`` and ``model.pcov_``.

        It is best to refer to these docs to understand the methods being wrapped:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        Args:
            model_func (Callable[[], np.array], optional): The function you wish to model. Defaults to None.
            p0 (Optional[List[float]], optional): The intial guess. Defaults to None.
            bounds (Union[ Tuple[np.dtype, np.dtype], List[Tuple[np.dtype, np.dtype]],
                Callable[ [], List[Tuple[np.dtype, np.dtype]]] ], optional): Bounds for trf. Defaults to (-np.inf, np.inf).
            method (str, optional): The curve fit method. Defaults to 'trf'.
            jac (Union[str, Callable[[np.array, Any], np.array], None ], optional): The jacobian matrix.
                If one is not provided then curve_fit will calculate it. Defaults to None.
            lsq_kwargs (dict, optional): Extra arguments for underlying lsq implementation. See `scipy.optimize.least_squares`. Defaults to None.
        """
        self.model_func = model_func
        self.p0 = p0
        self.bounds = bounds
        self.method = method
        self.jac = jac
        self.lsq_kwargs = lsq_kwargs if lsq_kwargs is not None else {}

    def fit(
        self,
        X: np.array,
        y: np.array = None,
        sigma: Optional[np.array] = None,
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
        X, y = check_X_y(X, y)  # pass the sklearn estimator dimensionality check

        if callable(self.bounds):  # we allow bounds to be a callable
            bounds = self.bounds(X)
        else:
            bounds = self.bounds

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

        self.popt_ = popt  # set optimized parameters on the instance
        self.pcov_ = pcov  # set optimzed covariances on the instance
        self.name_ = (
            self.model_func.__name__
        )  # name of func in case we are trying to fit multiple funcs in a Pipeline

        return self

    def predict(self, X: np.array) -> np.array:
        """Predict the target y values given X features using the best fit
        model (model_func) and best fit model parameters (popt)

        Args:
            X (np.array): The X matrix

        Returns:
            np.array: The predicted y values
        """
        check_is_fitted(self, ["popt_", "pcov_", "name_"])
        X = check_array(X)

        return self.model_func(X, *self.popt_)


class EnergyChangepointEstimator(BaseEstimator, RegressorMixin):
    """A container object for a changepoint model. After a model is fit you can access scores and
    load calculations via propeties.
    """

    def __init__(self, model: Optional[ModelFunction] = None):
        self.model = model

    @classmethod
    def sort_X_y(cls, X: NByOneNDArray, y: OneDimNDArray):
        X, y, _ = argsort_1d_idx(X, y)
        return X, y

    @classmethod
    def fit_many(
        cls,
        models: List[ModelFunction],
        X: NByOneNDArray,
        y: OneDimNDArray,
        sigma: Optional[OneDimNDArray] = None,
        absolute_sigma: Optional[bool] = None,
        sort: bool = True,
        fail_silently: bool = True,
        **estimator_kwargs,
    ) -> Generator[Tuple[str, Optional["EnergyChangepointEstimator"]], None, None]:
        if sort:
            X, y = cls.sort_X_y(X, y)

        for m in models:
            est = cls(m)
            try:
                yield est.name, est.fit(X, y, sigma, absolute_sigma, **estimator_kwargs)
            except (
                Exception
            ):  # XXX this is bad... need to figure out exactly what to catch here .. prob LinAlgError? or something from skl...
                if fail_silently:
                    yield m.name, None
                else:
                    raise

    @classmethod
    def fit_one(
        cls,
        model: ModelFunction,
        X: NByOneNDArray,
        y: OneDimNDArray,
        sigma: Optional[OneDimNDArray] = None,
        absolute_sigma: Optional[bool] = None,
        sort: bool = True,
        fail_silently: bool = True,
        **estimator_kwargs,
    ) -> Tuple[str, Optional["EnergyChangepointEstimator"]]:
        """Fits a single model. Will sort data if needed and optionally fail silently.

        Args:
            model (AbstractEnergyParameterModel): [description]
            data (CurvefitEstimatorData): [description]
            sort (bool, optional): [description]. Defaults to True.

        Returns:
            Optional[AbstractEnergyParameterModel]: [description]
        """
        if sort:
            X, y = cls.sort_X_y(X, y)

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
        X: NByOneNDArray,
        y: OneDimNDArray,
        sigma: Optional[OneDimNDArray] = None,
        absolute_sigma: Optional[bool] = None,
        **estimator_kwargs,
    ) -> "EnergyChangepointEstimator":
        """This is wrapper around CurvefitEstimator.fit and allows interoperability with sklearn
        NOTE: THIS METHOD DOES NOT SORT THE DATA! Input data should be sorted appropriately beforehand. You can
        use changepointmodel.core.utils.argsort_1d for standard changepoint model data.

        Args:
            data (CurvefitEstimatorInputData): [description]
            reshape (Optional[Tuple[int]], optional): [description]. Defaults to None.
        """
        self.estimator_ = CurvefitEstimator(
            model_func=self.model.f, bounds=self.model.bounds, **estimator_kwargs
        )
        self.pred_y_ = self.estimator_.fit(X, y, sigma, absolute_sigma).predict(
            X
        )  # call estimator fit  # XXX scipy/skl error in our own here.
        self.X_, self.y_ = (
            self.estimator_.X_,
            self.estimator_.y_,
        )  # XXX I think these need to be here for interop with skl cv
        self.sigma_ = sigma
        self.absolute_sigma_ = absolute_sigma

        return self

    def predict(self, X: NByOneNDArray) -> OneDimNDArray:
        """Proxy a call to estimator.predict in order to use the model to generate a changepoint model with
        different X vals on the fit estimator. This also allows interoperability with sklearn.

        Args:
            data (CurvefitEstimatorRegressorData): [description]

        Returns:
            np.array : predicted y values
        """
        check_is_fitted(self)
        return self.estimator_.predict(X)

    def adjust(self, other: "EnergyChangepointEstimator") -> OneDimNDArray:
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
    def name(self):
        if self.model is None:
            raise ValueError("Cannot access name of model that is not set.")
        return self.model.name

    @property
    @check_not_fitted
    def X(self) -> NByOneNDArray:
        return self.X_

    @property
    @check_not_fitted
    def y(self) -> OneDimNDArray:
        return self.y_

    @property
    @check_not_fitted
    def coeffs(self) -> Tuple[float, ...]:  # tuple
        return self.estimator_.popt_

    @property
    @check_not_fitted
    def cov(self) -> Tuple[float, ...]:
        return self.estimator_.pcov_

    @property
    @check_not_fitted
    def pred_y(self) -> OneDimNDArray:
        return self.pred_y_

    @property
    @check_not_fitted
    def sigma(self) -> Optional[OneDimNDArray]:
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
        return np.sum(self.pred_y_)

    @check_not_fitted
    def total_y(self) -> float:
        return np.sum(self.estimator_.y_)

    @check_not_fitted
    def len_y(self) -> int:
        return len(self.estimator_.y_)
