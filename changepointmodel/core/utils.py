from .pmodels import EnergyParameterModelCoefficients, ParameterModelFunction
from typing import Tuple, List
import numpy as np
from .nptypes import (
    CpModelXArray,
    OneDimNDArrayField,
    ArgSortRetType,
    AnyByAnyNDArrayField,
)


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
    arr: AnyByAnyNDArrayField, original_order: List[int]
) -> OneDimNDArrayField:
    """flattens and resorts numpy array back to its original order.

    Args:
        arr (nptypes.AnyByAnyNDArrayField): _description_
        original_order (List[int]): _description_

    Returns:
        nptypes.OneDimNDArrayField: _description_
    """
    out = arr.flatten()  # this would flatten X (oat)
    unsort_index = np.argsort(original_order)
    return out[unsort_index]


def parse_coeffs(
    model: ParameterModelFunction, coeffs: Tuple[float, ...]
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
