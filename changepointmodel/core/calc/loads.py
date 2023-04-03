""" Loads calculated from changepoint models. This is essentially AUC calculations.
"""

import numpy as np
from ..nptypes import OneDimNDArray


# XXX possibly remove this... possibly use absolute value here .. tynabot do research...
def _positive_sum(arr: OneDimNDArray[np.float64]) -> float:
    return np.sum((arr > 0) * arr)  # type: ignore


def _heating_predicted_load(
    X: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    yint: float,
    changepoint: float = -np.inf,
) -> OneDimNDArray[np.float64]:
    return (X < changepoint) * (pred_y - yint)


def _cooling_predicted_load(
    X: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    yint: float,
    changepoint: float = -np.inf,
) -> OneDimNDArray[np.float64]:
    return (X > changepoint) * (pred_y - yint)


def heatload(
    X: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    yint: float,
    changepoint: float = np.inf,
) -> float:
    """Calculate the heatload (auc) for both linear and changepoint models.

    Args:
        X (OneDimNDArray): The X array.
        pred_y (OneDimNDArray): The predicted y array
        yint (float): The y intercept.
        changepoint (Optional[float]): The model's changepoint if it has one. Defaults to np.inf to handle linear models.

    Returns:
        float: The heatload
    """
    predicted_loads = _heating_predicted_load(X, pred_y, yint, changepoint)
    return _positive_sum(predicted_loads)


def coolingload(
    X: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    yint: float,
    changepoint: float = -np.inf,
) -> float:
    """The cooling load (auc) for both linear and changepoint models.

    Args:
        X (OneDimNDArray): The X array.
        pred_y (OneDimNDArray): The predicted y array.
        yint (float): The y intercept.
        changepoint (float, optional): The changepoint if it has one. Defaults to -np.inf to handle linear models.

    Returns:
        float: [description]
    """
    predicted_loads = _cooling_predicted_load(X, pred_y, yint, changepoint)
    return _positive_sum(predicted_loads)


def baseload(total_consumption: float, heatload: float, coolingload: float) -> float:
    """The baseload calculated as the total_consumption minus loads.

    Args:
        total_consumption (float): The total consumption for the facility.
        heatload (float): The heating load.
        coolingload (float): The cooling load.

    Returns:
        float: The baseload.
    """
    return (
        total_consumption - heatload - coolingload
    )  # XXX if this is negative should we zero it out? Probably it should never be negative because energy...
