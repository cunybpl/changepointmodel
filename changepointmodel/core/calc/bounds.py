""" Dynamically calculates trf bounds for changepoint models where we approximately expect them to fit energy data based on 
the values of the X array.

Tuple return types correspond directly to coeffs in the method signature.Details on the structure of the bounds 
tuples can be read in the docs for `scipy.optimize.curve_fit`.
"""

from typing import Tuple, Union, Callable
import numpy as np
from ..nptypes import OneDimNDArray, NByOneNDArray

BoundTuple = Tuple[Tuple[float, ...], Tuple[float, ...]]
TwoParameterBoundary = Tuple[float, float]
ThreeParameterBoundary = Tuple[float, float, float]
FourParameterBoundary = Tuple[float, float, float, float]
FiveParameterBoundary = Tuple[float, float, float, float, float]

OpenBoundCallable = Callable[
    [Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]], BoundTuple
]


def twop(*args, **kwargs) -> Tuple[TwoParameterBoundary, TwoParameterBoundary]:  # type: ignore
    """Energy bound for a twop (linear) model. Essentially returns a constant but we need this to
    conform to the Bounds interface.

    Returns:
        Tuple[TwoParameterBoundary, TwoParameterBoundary]: The returned bounds for scipy.optimize.curve_fit
    """
    return ((0, -np.inf), (np.inf, np.inf))


def threepc(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[ThreeParameterBoundary, ThreeParameterBoundary]:
    """A threepc boundary for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[ThreeParameterBoundary, ThreeParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    min_cp = X[int(len(X) / 4)]
    max_cp = X[int(3 * len(X) / 4)]
    return ((0, 0, min_cp), (np.inf, np.inf, max_cp))


def threeph(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[ThreeParameterBoundary, ThreeParameterBoundary]:
    """A threeph boundary for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[ThreeParameterBoundary, ThreeParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    min_cp = X[int(len(X) / 4)]
    max_cp = X[int(3 * len(X) / 4)]
    return ((0, -np.inf, min_cp), (np.inf, 0, max_cp))


def fourp(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[FourParameterBoundary, FourParameterBoundary]:
    """A fourp boundary for energy data

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[FourParameterBoundary, FourParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    min_cp = X[int(len(X) / 4)]
    max_cp = X[int(3 * len(X) / 4)]
    return ((0, -np.inf, -np.inf, min_cp), (np.inf, np.inf, np.inf, max_cp))


def fivep(
    X: Union[OneDimNDArray[np.float64], NByOneNDArray[np.float64]]
) -> Tuple[FiveParameterBoundary, FiveParameterBoundary]:
    """A fivep boundary for energy data.

    Args:
        X (Union[OneDimNDArray,NByOneNDArray]): A numpy X array. NByOneNDArray's will be squeezed internally.

    Returns:
        Tuple[FiveParameterBoundary, FiveParameterBoundary]: Resulting bounds tuples.
    """
    X = X.squeeze()
    min_cp1 = X[int((2 / 8) * len(X))]
    max_cp1 = X[int((3 / 8) * len(X))]
    min_cp2 = X[int((5 / 8) * len(X))]
    max_cp2 = X[int((6 / 8) * len(X))]
    return ((0, -np.inf, 0, min_cp1, min_cp2), (np.inf, 0, np.inf, max_cp1, max_cp2))
