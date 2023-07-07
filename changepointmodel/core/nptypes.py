""" Provides numpy types for various functionality throughout the library as well as pydantic fields 
that are used for validating input and converting it from python primitives or arrays.

This can be treated as more or less a private module.     
"""
import numpy as np
from typing import Union, Tuple, TypeVar, Annotated, Literal, Any

import numpy.typing as npt

# https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype


Dtype = TypeVar("Dtype", bound=np.generic)
AnyByAnyNDArray = npt.NDArray[Dtype]
NByOneNDArray = Annotated[npt.NDArray[Dtype], Literal["N", 1]]
OneDimNDArray = Annotated[npt.NDArray[Dtype], Literal[1]]

SklScoreReturnType = Union[float, OneDimNDArray[np.float64]]


class AnyByAnyNDArrayField(np.ndarray):  # type: ignore
    @classmethod
    def __get_validators__(cls):  # type: ignore
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> AnyByAnyNDArray[np.float64]:
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 2:
            raise ValueError("Shape of data should be M x n")
        return arr

    @classmethod
    def __modify_schema__(cls, field_val: Any) -> None:
        field_val.update(type="array")
        field_val.update(items={"type": "array", "items": {"type": "number"}})


class NByOneNDArrayField(np.ndarray):  # type: ignore
    @classmethod
    def __get_validators__(cls):  # type: ignore
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> NByOneNDArray[np.float64]:
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 2:
            raise ValueError("Shape of data should be M x n")
        if arr.shape[1] != 1:
            raise ValueError(f"Second dimension must be of size 1, got {arr.shape[1]}")
        return arr

    @classmethod
    def __modify_schema__(cls, field_val: Any) -> None:
        field_val.update(type="array")
        field_val.update(items={"type": "array", "items": {"type": "number"}})


class OneDimNDArrayField(np.ndarray):  # type: ignore
    @classmethod
    def __get_validators__(cls):  # type: ignore
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> OneDimNDArray[np.float64]:
        arr = np.array(v, dtype=float)
        if len(arr.shape) != 1:
            raise ValueError("Shape of data should be One dimension")
        return arr

    @classmethod
    def __modify_schema__(cls, field_val: Any) -> None:
        field_val.update(type="array")
        field_val.update(items={"type": "number"})


CpModelXArray = Union[OneDimNDArrayField, AnyByAnyNDArrayField]
Ordering = npt.NDArray[np.intp]
ArgSortRetType = Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Ordering]
