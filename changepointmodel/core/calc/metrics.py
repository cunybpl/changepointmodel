""" Scoring metrics. These are implemented as wrappers around sklearn.metrics.
The scorer module forwards calls to these implementations.
"""
from typing import Union
import numpy as np
from sklearn import metrics as sklmetrics  # type: ignore

from ..nptypes import OneDimNDArray

# XXX use nptyping here.


def _cvrmse_from_rmse(rmse: float, y: OneDimNDArray) -> float:
    return rmse / np.mean(y)


def r2_score(
    y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs
) -> Union[float, OneDimNDArray[float]]:
    """A wrapper around sklearn.metrics.r2_score. Returns the r2 score for predicted y values.

    Args:
        y (OneDimNDArray): The original y values.
        y_pred (OneDimNDArray): The predicted y values.

    Returns:
        Union[float, OneDimNDArray[float]]: The R2 scores or array of r2 scores if weigted.
    """
    return sklmetrics.r2_score(y, y_pred, **kwargs)


def rmse(
    y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs
) -> Union[float, OneDimNDArray[float]]:
    """The root mean square error using sklearn.metrics.mean_squared_error. We set squared=False
    according to the sklearn docs to get rmse.

    Args:
        y (OneDimNDArray): The original y values.
        y_pred (OneDimNDArray): The predicted y values.

    Returns:
        Union[float, OneDimNDArray[float]]: The RMSE or array of scores if weighted.
    """
    return sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)


def cvrmse(
    y: OneDimNDArray, y_pred: OneDimNDArray, **kwargs
) -> Union[float, OneDimNDArray[float]]:
    """Calculates the cvrmse from rmse.

    Args:
        y (OneDimNDArray): The original y values.
        y_pred (OneDimNDArray): The predicted y values.

    Returns:
        Union[float, OneDimNDArray[float]]: The CVRMSE or array of scores if weighted.
    """
    rmse = sklmetrics.mean_squared_error(y, y_pred, squared=False, **kwargs)
    return _cvrmse_from_rmse(rmse, y)
