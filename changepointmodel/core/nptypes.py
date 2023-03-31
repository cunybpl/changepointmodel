""" Provides numpy types for various functionality throughout the library as well as pydantic fields 
that are used for validating input and converting it from python primitives or arrays.

This can be treated as more or less a private module.     
"""
import numpy as np
from typing import Any, List, Union, Tuple
from nptyping import NDArray

AnyByAnyNDArray = NDArray[(Any, ...), float]
NByOneNDArray = NDArray[
    (
        Any,
        1,
    ),
    float,
]  # [[1.], [2.], [3.], ...]
OneDimNDArray = NDArray[(Any,), float]  # [ 1., 2., 3., ...]


# pydantic aware numpy types.


class AnyByAnyNDArrayField(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 2:
            raise ValueError("Shape of data should be M x n")
        return arr

    @classmethod
    def __modify_schema__(cls, field_val):
        field_val.update(type="array")
        field_val.update(items={"type": "array", "items": {"type": "number"}})


class NByOneNDArrayField(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 2:
            raise ValueError("Shape of data should be M x n")
        if arr.shape[1] != 1:
            raise ValueError(f"Second dimension must be of size 1, got {arr.shape[1]}")
        return arr

    @classmethod
    def __modify_schema__(cls, field_val):
        field_val.update(type="array")
        field_val.update(items={"type": "array", "items": {"type": "number"}})


class OneDimNDArrayField(np.ndarray):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 1:
            raise ValueError("Shape of data should be One dimension")
        return arr

    @classmethod
    def __modify_schema__(cls, field_val):
        field_val.update(type="array")
        field_val.update(items={"type": "number"})


CpModelXArray = Union[OneDimNDArrayField, AnyByAnyNDArrayField]
Ordering = List[int]
ArgSortRetType = Tuple[AnyByAnyNDArrayField, OneDimNDArrayField, Ordering]
