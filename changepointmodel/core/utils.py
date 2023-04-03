from .pmodels import (
    ParameterModelFunction,
    EnergyParameterModelT,
    EnergyParameterModelCoefficients,
    ParamaterModelCallableT,
)
from typing import Tuple, List
import numpy as np
from .nptypes import (
    CpModelXArray,
    OneDimNDArrayField,
    ArgSortRetType,
    AnyByAnyNDArrayField,
)

import numpy.typing as npt


def argsort_1d_idx(X: CpModelXArray, y: OneDimNDArrayField) -> ArgSortRetType:
    """Sort a numpy array and return an ordering to be used later to unorder arrays.

    Args:
        X (CpModelXArray): _description_
        y (nptypes.OneDimNDArrayField): _description_

    Returns:
        ArgSortRetType: _description_
    """
    order = np.argsort(X.squeeze())
    return X[order], y[order], order


def unargsort_1d_idx(
    arr: npt.NDArray[np.float64], original_order: List[int]
) -> OneDimNDArrayField:
    """flattens and resorts numpy array back to its original order.

    Args:
        arr (nptypes.AnyByAnyNDArrayField): _description_
        original_order (List[int]): _description_

    Returns:
        nptypes.OneDimNDArrayField: _description_
    """
    out = arr.flatten()
    unsort_index = np.argsort(original_order)
    return out[unsort_index]  # type: ignore


def parse_coeffs(
    model: ParameterModelFunction[ParamaterModelCallableT, EnergyParameterModelT],
    coeffs: Tuple[float, ...],
) -> EnergyParameterModelCoefficients:
    """Given an ParameterModelFunction and raw coefficients tuple from CurvefitEstimator.fit
    will return an EnerguParameterModelCoefficients accessor object.

    Args:
        model (EnergyParameterModel): [description]
        coeffs (Tuple[float, ...]): [description]

    Returns:
        EnergyParameterModelCoefficients: [description]
    """
    return model.parse_coeffs(coeffs)
