from .pmodels import (
    ParameterModelFunction,
    EnergyParameterModelT,
    EnergyParameterModelCoefficients,
    ParamaterModelCallableT,
)
from typing import Tuple

from .nptypes import (
    ArgSortRetType,
    Ordering,
)

import numpy as np
import numpy.typing as npt


def argsort_1d_idx(
    X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> ArgSortRetType:
    """Sort a numpy array and return an ordering to be used later to unorder arrays.

    Args:
        X (npt.NDArray[np.float64]): _description_
        y (npt.NDArray[np.float64]): _description_

    Returns:
        ArgSortRetType: A tuple of sorted X, y and order
    """
    order = np.argsort(X.squeeze())
    return X[order], y[order], order


def unargsort_1d_idx(
    arr: npt.NDArray[np.float64], original_order: Ordering
) -> npt.NDArray[np.float64]:
    """flattens and resorts a numpy array back to its original order.

    Args:
        arr (npt.NDArray[np.float64]): The previously sorted array.
        original_order (Ordering): The original ordering evaluated from argort_1d_idx

    Returns:
        npt.NDArray[np.float64]: The unsorted array.
    """
    out = arr.flatten()
    unsort_index = np.argsort(original_order)
    return out[unsort_index]


def parse_coeffs(
    model: ParameterModelFunction[ParamaterModelCallableT, EnergyParameterModelT],
    coeffs: Tuple[float, ...],
) -> EnergyParameterModelCoefficients:
    """Given an ParameterModelFunction and raw coefficients tuple from CurvefitEstimator.fit
    will return an EnergyParameterModelCoefficients accessor object. Essentially translates a raw
    tuple into a defined type.

    Args:
        model (ParameterModelFunction): The model function container instance.
        coeffs (Tuple[float, ...]): A set of coeffs from curve_fit.

    Returns:
        EnergyParameterModelCoefficients: A well defined type of coeffs.
    """
    return model.parse_coeffs(coeffs)
