""" APIs for handling adjusted and normalized savings as per ashrae.
"""
from dataclasses import dataclass
import numpy as np
from .nptypes import NByOneNDArray, OneDimNDArray
from .estimator import EnergyChangepointEstimator
from .calc import savings as libsavings

from .scoring import Cvrmse
from typing import Optional, Tuple, TypeVar

import numpy.typing as npt

_cvrmse_score = Cvrmse()

import abc
from .pmodels import ParamaterModelCallableT, EnergyParameterModelT


# I have to pass the pydantic ndarray objects as fields so
# that we can validate the data later.


def _get_adjusted(
    pre: EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT],
    post: EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT],
) -> OneDimNDArray[np.float64]:
    return pre.adjust(post)


@dataclass
class AdjustedSavingsResult(object):
    adjusted_y: npt.NDArray[np.float64]
    total_savings: float
    average_savings: float
    percent_savings: float
    percent_savings_uncertainty: float


@dataclass
class NormalizedSavingsResult(object):
    normalized_y_pre: npt.NDArray[np.float64]
    normalized_y_post: npt.NDArray[np.float64]
    total_savings: float
    average_savings: float
    percent_savings: float
    percent_savings_uncertainty: float


class IAdjustedSavingsCalculator(abc.ABC):
    @abc.abstractmethod
    def save(
        self,
        pre: EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT],
        post: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> AdjustedSavingsResult:
        ...


class INormalizedSavingsCalculator(abc.ABC):
    @abc.abstractmethod
    def save(
        self,
        pre: EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT],
        post: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> NormalizedSavingsResult:
        ...


class AbstractAdjustedSavingsCalculator(IAdjustedSavingsCalculator):
    def __init__(
        self, confidence_interval: float = 0.80, scalar: Optional[float] = None
    ):
        """A Savings model calculates either adjusted or weather normalized savings
        based on ashrae formulas and methodology. Should be used in the context of option-c
        reporting with changepoint models.

        Scalar should be used if your data is in avg per day per month and you need to scale it back out.
        For this scenario set scalar to 30.437.

        Args:
            confidence_interval (float, optional): The confidence interval to be used for the calculations. Defaults to 0.80.
            scalar (float, optional): Value to scale by. Use 30.437 to scale from per-day to total month!!
        """
        self._confidence_interval = confidence_interval
        self._scalar = 1 if scalar is None else scalar

    @property
    def confidence_interval(self) -> float:
        return self._confidence_interval

    @abc.abstractmethod
    def _savings(
        self,
        gross_adjusted_pred_y: float,
        gross_post_y: float,
        pre_cvrmse: float,
        pre_p: int,
        pre_n: int,
        post_n: int,
        confidence_interval: float,
    ) -> Tuple[float, float, float, float]:
        """Override this method in a subclass to provide a savings calculation.

        Args:
            gross_adjusted_pred_y (float): _description_
            gross_post_y (float): _description_
            pre_cvrmse (float): _description_
            pre_p (int): _description_
            pre_n (int): _description_
            post_n (int): _description_
            confidence_interval: _description_

        Returns:
            Tuple[float, float, float, float]: A tuple of total_savings, average_savings, percent_savings and percent_savings_uncertainity.
        """
        ...

    def save(
        self,
        pre: EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT],
        post: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> AdjustedSavingsResult:
        """Controller method for calculated AdjustedSavings values and uncertainiies.

        Args:
            pre (EnergyChangepointEstimator): A fitted model for the pre retrofit period
            post (EnergyChangepointEstimator): A fitted model for the post retrofit period

        Returns:
            AdjustedSavingsResult: The AdjustedSavings result.
        """

        # adding scalar ... only pred_y gets adjusted by scalar here... not in cvrmse
        adjusted_pred_y = _get_adjusted(pre, post) * self._scalar
        gross_adjusted_y = np.sum(adjusted_pred_y)
        gross_post_y = np.sum(post.pred_y * self._scalar)

        pre_cvrmse = _cvrmse_score(pre.y, pre.pred_y)
        pre_p = len(pre.coeffs)

        pre_n = pre.len_y()
        post_n = post.len_y()

        savings = self._savings(
            float(gross_adjusted_y),
            float(gross_post_y),
            float(pre_cvrmse),
            pre_p,
            pre_n,
            post_n,
            self._confidence_interval,
        )

        (
            total_savings,
            average_savings,
            percent_savings,
            percent_savings_uncertainty,
        ) = savings
        return AdjustedSavingsResult(
            adjusted_pred_y,
            total_savings,
            average_savings,
            percent_savings,
            percent_savings_uncertainty,
        )


class AshraeAdjustedSavingsCalculator(AbstractAdjustedSavingsCalculator):
    def _savings(
        self,
        gross_adjusted_pred_y: float,
        gross_post_y: float,
        pre_cvrmse: float,
        pre_p: int,
        pre_n: int,
        post_n: int,
        confidence_interval: float = 0.8,
    ) -> Tuple[float, float, float, float]:
        return libsavings.adjusted(
            float(gross_adjusted_pred_y),
            float(gross_post_y),
            float(pre_cvrmse),
            pre_p,
            pre_n,
            post_n,
            confidence_interval,
        )


class AbstractNormalizedSavingsCalculator(INormalizedSavingsCalculator):
    def __init__(
        self,
        X_norms: NByOneNDArray[np.float64],
        confidence_interval: float = 0.80,
        scalar: Optional[float] = None,
    ):
        """The Normalized savings calculations provide pre and post X arrays. These are used within the context
        of weather normalized savings for option-c retrofits.

        Args:
            X_norms (NByOneNDArray): Normalized X data for pre-retrofit and post-retrofit related normalized calculation.
            confidence_interval (float, optional): The confidence interval for the uncertainity calculations. Defaults to 0.80.
        """
        self._X_norms = X_norms
        self._confidence_interval = confidence_interval
        self._scalar = 1 if scalar is None else scalar

    @property
    def X_norms(self) -> NByOneNDArray[np.float64]:
        return self._X_norms

    @property
    def confidence_interval(self) -> float:
        return self._confidence_interval

    @abc.abstractmethod
    def _savings(
        self,
        gross_normalized_pred_y_pre: float,
        gross_normalized_pred_y_post: float,
        pre_cvrmse: float,
        post_cvrmse: float,
        pre_n: int,
        post_n: int,
        pre_p: int,
        post_p: int,
        n_norm: int,
        confidence_interval: float,
    ) -> Tuple[float, float, float, float]:
        ...

    def save(
        self,
        pre: EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT],
        post: EnergyChangepointEstimator[
            ParamaterModelCallableT, EnergyParameterModelT
        ],
    ) -> NormalizedSavingsResult:
        """The controller method for the normalized savings calculation.

        Args:
            pre (EnergyChangepointEstimator): The fitted pre retrofit model.
            post (EnergyChangepointEstimator): The fitted post retrofit model.

        Returns:
            NormalizedSavingsResult: The result of the normalized savings calculation
        """

        # setup
        normalized_pred_y_pre = pre.predict(self._X_norms) * self._scalar
        normalized_pred_y_post = post.predict(self._X_norms) * self._scalar

        gross_normalized_pred_y_pre = np.sum(normalized_pred_y_pre)
        gross_normalized_pred_y_post = np.sum(normalized_pred_y_post)

        pre_cvrmse = _cvrmse_score(pre.y, pre.pred_y)
        post_cvrmse = _cvrmse_score(post.y, post.pred_y)

        pre_n = pre.len_y()
        post_n = post.len_y()

        pre_p = len(pre.coeffs)
        post_p = len(post.coeffs)

        n_norm = len(self._X_norms)

        savings = self._savings(
            float(gross_normalized_pred_y_pre),
            float(gross_normalized_pred_y_post),
            float(pre_cvrmse),
            float(post_cvrmse),
            pre_n,
            post_n,
            pre_p,
            post_p,
            n_norm,
            self._confidence_interval,
        )

        (
            total_savings,
            average_savings,
            percent_savings,
            percent_savings_uncertainty,
        ) = savings

        return NormalizedSavingsResult(
            normalized_pred_y_pre,
            normalized_pred_y_post,
            total_savings,
            average_savings,
            percent_savings,
            percent_savings_uncertainty,
        )


class AshraeNormalizedSavingsCalculator(AbstractNormalizedSavingsCalculator):
    def _savings(
        self,
        gross_normalized_pred_y_pre: float,
        gross_normalized_pred_y_post: float,
        pre_cvrmse: float,
        post_cvrmse: float,
        pre_n: int,
        post_n: int,
        pre_p: int,
        post_p: int,
        n_norm: int,
        confidence_interval: float,
    ) -> Tuple[float, float, float, float]:
        return libsavings.weather_normalized(
            gross_normalized_pred_y_pre,
            gross_normalized_pred_y_post,
            pre_cvrmse,
            post_cvrmse,
            pre_n,
            post_n,
            pre_p,
            post_p,
            n_norm,
            confidence_interval,
        )
