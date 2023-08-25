"""Defines types to use as inputs to CurvefitEstimator. This is agnostic of function since 
NumpyArray can be n-dimensional
"""
import numpy as np
from typing import Any, Dict, Optional, Union

import pydantic
from .nptypes import (
    OneDimNDArray,
    NByOneNDArray,
    ArgSortRetType,
)

from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema, ConfigDict

from typing import Annotated, Any

from .utils import argsort_1d_idx


def _validate_one_dim(v: Any) -> OneDimNDArray[np.float64]:
    arr = np.array(v, dtype=float)
    if len(arr.shape) != 1:
        raise ValueError("Shape of data should be One dimension")
    return arr


def _validate_n_by_one_dim(v: Any) -> NByOneNDArray[np.float64]:
    arr = np.array(v, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)

    if len(arr.shape) != 2:
        raise ValueError("Shape of data should be M x n")

    if arr.shape[1] != 1:
        raise ValueError(f"Second dimension must be of size 1, got {arr.shape[1]}")

    return arr


from typing import List

OneDimNDArrayField = Annotated[
    OneDimNDArray[np.float64],
    BeforeValidator(_validate_one_dim),
    PlainSerializer(lambda x: x.tolist(), return_type=List),
    WithJsonSchema({"type": "array", "items": {"type": "number"}}),
]


NByOneNDArrayField = Annotated[
    NByOneNDArray[np.float64],
    BeforeValidator(_validate_n_by_one_dim),
    PlainSerializer(lambda x: x.tolist(), return_type=List),
    WithJsonSchema(
        {"type": "array", "items": {"items": {"type": "number"}, "type": "array"}}
    ),
]


class CurvefitEstimatorDataModel(pydantic.BaseModel):
    X: NByOneNDArrayField
    y: OneDimNDArrayField
    sigma: Optional[OneDimNDArrayField] = None
    absolute_sigma: Optional[bool] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @pydantic.model_validator(mode="after")
    def validate_all(self) -> "CurvefitEstimatorDataModel":
        if len(self.X) != len(self.y):
            raise ValueError("X and y len must be the same.")

        if self.sigma is not None and len(self.sigma) != len(self.X):
            raise ValueError("len of sigma must match len X and y")

        return self

    def sorted_X_y(self) -> ArgSortRetType:
        """Helper to sort X and y. Also returns the original idx ordering to reverse.
        X will be reshaped as 2D array in order to work with sklearn dimenisonality.

        Returns:
            ArgSortRetType: Sorted X, y and original ordering index
        """
        return argsort_1d_idx(self.X, self.y)
