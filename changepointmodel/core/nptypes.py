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


Ordering = npt.NDArray[np.intp]
ArgSortRetType = Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], Ordering]
