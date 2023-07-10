""" Scoring metrics. These are implemented as wrappers around sklearn.metrics.
The scorer module forwards calls to these implementations.
"""
from typing import Union
import numpy as np
from sklearn import metrics as sklmetrics  # type: ignore
from ..nptypes import OneDimNDArray

from typing import Dict, Any


def r2_score(
    y: OneDimNDArray[np.float64],
    y_pred: OneDimNDArray[np.float64],
    **kwargs: Dict[str, Any]
) -> Union[float, OneDimNDArray[np.float64]]:
    """A wrapper around sklearn.metrics.r2_score. Returns the r2 score for predicted y values.

    Args:
        y (OneDimNDArray): The original y values.
        y_pred (OneDimNDArray): The predicted y values.

    Returns:
        Union[float, OneDimNDArray[float]]: The R2 scores or array of r2 scores if weigted.
    """
    return sklmetrics.r2_score(y, y_pred, **kwargs)  # type: ignore


def rmse(
    y: OneDimNDArray[np.float64],
    y_pred: OneDimNDArray[np.float64],
    **kwargs: Dict[str, Any]
) -> Union[float, OneDimNDArray[np.float64]]:
    """The root mean square error using sklearn.metrics.mean_squared_error. We set squared=False
    according to the sklearn docs to get rmse.

    Args:
        y (OneDimNDArray): The original y values.
        y_pred (OneDimNDArray): The predicted y values.

    Returns:
        Union[float, OneDimNDArray[float]]: The RMSE or array of scores if weighted.
    """
    return sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)  # type: ignore


def cvrmse(
    y: OneDimNDArray[np.float64],
    y_pred: OneDimNDArray[np.float64],
    **kwargs: Dict[str, Any]
) -> Union[float, OneDimNDArray[np.float64]]:
    """Calculates the cvrmse from rmse.

    Args:
        y (OneDimNDArray): The original y values.
        y_pred (OneDimNDArray): The predicted y values.

    Returns:
        Union[float, OneDimNDArray[float]]: The CVRMSE or array of scores if weighted.
    """
    rmse = sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)
    return rmse / np.mean(y)  # type: ignore


def adjusted_r2_score(
    y: OneDimNDArray[np.float64], y_pred: OneDimNDArray[np.float64], p: int = 1
) -> Union[float, OneDimNDArray[np.float64]]:
    """Calculate the adjusted r2

    Args:
        y (OneDimNDArray[np.float64]): original y values.
        y_pred (OneDimNDArray[np.float64]): predicted y values
        p (int, optional): Number of parameters in the model. Defaults to 1.

    Returns:
        Union[float, OneDimNDArray[np.float64]]: The adjusted_r2 or array of scores if weighted
    """
    r2 = r2_score(y, y_pred)
    n = len(y)
    return 1 - ((n - 1) / n - p) * (1 - r2)
