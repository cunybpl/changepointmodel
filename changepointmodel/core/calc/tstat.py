import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional
from changepointmodel.core.nptypes import OneDimNDArray

# Return types

# SingleSlopeTStat = float
# DoubleSlopeTStat = Tuple[float, float]

HeatingCoolingTStatResult = Tuple[Optional[float], Optional[float]]


# Helpers
def _std_error(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> float:
    sse = np.sum((y - y_pred) ** 2)
    n = np.sqrt(sse / (len(y) - 2))
    d = np.sqrt(np.sum((x - np.mean(x)) ** 2))
    return n / d  # type: ignore


def _get_array_right(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    cp: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    position = np.where(x >= cp)
    y_out = y[position]
    y_pred_out = y_pred[position]
    x_out = x[position]
    return x_out, y_out, y_pred_out


def _get_array_left(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
    cp: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    position = np.where(x <= cp)
    y_out = y[position]
    y_pred_out = y_pred[position]
    x_out = x[position]
    return x_out, y_out, y_pred_out


# Main functions
def twop(
    X: OneDimNDArray[np.float64],
    y: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    slope: float,
) -> HeatingCoolingTStatResult:
    tstat = slope / _std_error(X, y, pred_y)
    if slope <= 0:
        return tstat, None
    else:
        return None, tstat


def threepc(
    X: OneDimNDArray[np.float64],
    y: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    slope: float,
    changepoint: float,
) -> HeatingCoolingTStatResult:
    _x, _y, _pred_y = _get_array_right(
        X,
        y,
        pred_y,
        changepoint,
    )
    return None, slope / _std_error(_x, _y, _pred_y)


def threeph(
    X: OneDimNDArray[np.float64],
    y: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    slope: float,
    changepoint: float,
) -> HeatingCoolingTStatResult:
    _x, _y, _pred_y = _get_array_left(
        X,
        y,
        pred_y,
        changepoint,
    )
    return slope / _std_error(_x, _y, _pred_y), None


def fourp(
    X: OneDimNDArray[np.float64],
    y: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    ls: float,
    rs: float,
    changepoint: float,
) -> HeatingCoolingTStatResult:
    xl, yl, pred_yl = _get_array_left(
        X,
        y,
        pred_y,
        changepoint,
    )

    xr, yr, pred_yr = _get_array_right(
        np.array(X), np.array(y), np.array(pred_y), changepoint
    )

    tl = ls / _std_error(xl, yl, pred_yl)
    tr = rs / _std_error(xr, yr, pred_yr)
    return tl, tr


def fivep(
    X: OneDimNDArray[np.float64],
    y: OneDimNDArray[np.float64],
    pred_y: OneDimNDArray[np.float64],
    ls: float,
    rs: float,
    lcp: float,
    rcp: float,
) -> HeatingCoolingTStatResult:
    xl, yl, pred_yl = _get_array_left(
        X,
        y,
        pred_y,
        lcp,
    )

    xr, yr, pred_yr = _get_array_right(
        np.array(X),
        np.array(y),
        np.array(pred_y),
        rcp,
    )

    tl = ls / _std_error(xl, yl, pred_yl)
    tr = rs / _std_error(xr, yr, pred_yr)
    return (tl, tr)
