"""Defines functions for energy changepoint models based on EnergyExplorer and related work.

These functions are fit using the Trust-Region-Reflexive Algo in `scipy.optimize.curve_fit` and 
conform to the specs for that interface.
"""

from ..nptypes import NByOneNDArray, OneDimNDArray
import numpy as np


def twop(
    X: NByOneNDArray[np.float64], yint: float, m: float
) -> OneDimNDArray[np.float64]:
    """A two parameter (linear) model.

    Args:
        X (NByOneNDArray): The X array
        yint (float): The yintercept
        m (float): The slope

    Returns:
        OneDimNDArray: The y array of calculated values.
    """
    return (m * X + yint).squeeze()


def threepc(
    X: NByOneNDArray[np.float64], yint: float, m: float, cp: float
) -> OneDimNDArray[np.float64]:
    """A three parameter changepoint function that models cooling.

    Args:
        X (NByOneNDArray): The X array.
        yint (float): The y intercept (baseload)
        m (float): The slope of the cooling line.
        cp (float): The changepoint where the slope deviates from zero.

    Returns:
        OneDimNDArray: The y array of calculated values.
    """
    return ((X < cp) * (yint) + (X >= cp) * (m * (X - cp) + yint)).squeeze()


def threeph(
    X: NByOneNDArray[np.float64], yint: float, m: float, cp: float
) -> OneDimNDArray[np.float64]:
    """A three parameter changepoint function that models heating.

    Args:
        X (NByOneNDArray): The X array.
        yint (float): The y intercept (baseload)
        m (float): The slope of the heating line.
        cp (float): The changepoint where the slope deviates from zero.

    Returns:
        OneDimNDArray: The y array of calculated values.
    """
    return ((X < cp) * (m * (X - cp) + yint) + (X >= cp) * (yint)).squeeze()  # type: ignore


def fourp(
    X: NByOneNDArray[np.float64], yint: float, m1: float, m2: float, cp: float
) -> OneDimNDArray[np.float64]:
    """A four parameter changepoint function that models simultaneous heating and cooling with a
    single point of inflection and no zero slope.

    Args:
        X (NByOneNDArray): The X array.
        yint (float): The yintercept of the changepoint
        m1 (float): The slope left of changepoint
        m2 (float): The slope right of changepoint
        cp (float): The point where both slopes converge.

    Returns:
        OneDimNDArray: The y array of calculated values.
    """
    return (  # type: ignore
        (X < cp) * (m1 * (X - cp) + yint) + (X >= cp) * (m2 * (X - cp) + yint)
    ).squeeze()


def fivep(
    X: NByOneNDArray[np.float64],
    yint: float,
    m1: float,
    m2: float,
    cp1: float,
    cp2: float,
) -> OneDimNDArray[np.float64]:
    """A five parameter changepoint function that models simultaneous heating and cooling with both
    a heating and cooling changepoint.

    Args:
        X (NByOneNDArray): The X array.
        yint (float): The yintercept (baseload)
        m1 (float): The left slope of the changepoint.
        m2 (float): The right slope of the changepoint.
        cp1 (float): The left changepoint.
        cp2 (float): The right changepoint.

    Returns:
        OneDimNDArray: The y array of calculated values.
    """
    return (  # type: ignore
        (X < cp1) * (m1 * (X - cp1) + yint)
        + ((X < cp2) & (X >= cp1)) * (yint)
        + (X >= cp2) * (m2 * (X - cp2) + yint)
    ).squeeze()
