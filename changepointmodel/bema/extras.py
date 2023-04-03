""" Extras module provides edge case checks for building energy use cases. These guards are included 
by default within a filter config object but can be switched off if needed.

They will filter out any models silently that do not conform to these conditions.
"""


import numpy as np
from typing import Iterator, List, Tuple
from .base import BemaChangepointResultContainers, BemaChangepointResultContainer
from changepointmodel.core.pmodels import ParamaterModelCallableT, EnergyParameterModelT

import numpy.typing as npt


def dpop(
    results: BemaChangepointResultContainers[
        ParamaterModelCallableT, EnergyParameterModelT
    ],
) -> Iterator[
    BemaChangepointResultContainer[ParamaterModelCallableT, EnergyParameterModelT]
]:
    """Checks if a models heating and cooling are within an established threshold that makes
    sense for building data.

    Args:
        results (BemaChangepointResultContainers): _description_

    Yields:
        Iterator[BemaChangepointResultContainer]: _description_
    """
    for result in results:
        r = result.result

        if r.name == "2P":
            yield result

        elif r.name == "3PC":
            assert r.coeffs.changepoints is not None
            len_x = len(r.input_data.X)
            threshold = len_x / 4
            coolnum = sum(
                r.input_data.X >= r.coeffs.changepoints[0]
            )  # ge in bema legacy
            if coolnum >= threshold and len_x - coolnum >= threshold:
                yield result

        elif r.name == "3PH":
            assert r.coeffs.changepoints is not None
            len_x = len(r.input_data.X)
            threshold = len_x / 4
            heatnum = sum(r.input_data.X <= r.coeffs.changepoints[0])
            if heatnum >= threshold and len_x - heatnum >= threshold:
                yield result

        elif r.name == "4P":
            assert r.coeffs.changepoints is not None
            len_x = len(r.input_data.X)
            threshold = len_x / 4
            heatnum = sum(r.input_data.X <= r.coeffs.changepoints[0])
            coolnum = sum(r.input_data.X > r.coeffs.changepoints[0])
            if coolnum >= threshold and heatnum >= threshold:
                yield result

        elif r.name == "5P":
            assert r.coeffs.changepoints is not None
            len_x = len(r.input_data.X)
            threshold = len_x / 4
            heatnum = sum(r.input_data.X <= r.coeffs.changepoints[0])
            coolnum = sum(
                r.input_data.X >= r.coeffs.changepoints[1]
            )  # inclusive aka ge in bemalegacy
            if (
                coolnum >= threshold
                and heatnum >= threshold
                and len_x - (heatnum + coolnum) >= threshold
            ):
                yield result


def shape(
    results: BemaChangepointResultContainers[
        ParamaterModelCallableT, EnergyParameterModelT
    ],
) -> Iterator[
    BemaChangepointResultContainer[ParamaterModelCallableT, EnergyParameterModelT]
]:
    """Checks that certain slopes of models conform to a shape expected from building energy data.

    Args:
        results (BemaChangepointResultContainers): _description_

    Yields:
        Iterator[BemaChangepointResultContainer]: _description_
    """
    for result in results:
        r = result.result
        if r.name == "2P":
            yield result

        if r.name == "3PC" and r.coeffs.slopes[0] > 0:
            yield result

        elif r.name == "3PH" and r.coeffs.slopes[0] < 0:
            yield result

        elif r.name == "4P" or r.name == "5P":
            ls, rs = r.coeffs.slopes
            if ls < 0 and rs > 0:  # should be V shape
                if abs(ls) > abs(rs):  # check the magnitude of the slopes
                    yield result


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


def tstat(
    results: BemaChangepointResultContainers[
        ParamaterModelCallableT, EnergyParameterModelT
    ],
) -> Iterator[
    BemaChangepointResultContainer[ParamaterModelCallableT, EnergyParameterModelT]
]:
    """Determines if slopes are statistically significant relevant to one another.

    Args:
        results (BemaChangepointResultContainers): _description_

    Yields:
        Iterator[BemaChangepointResultContainer]: _description_
    """
    for result in results:
        r = result.result

        if r.name == "2P":
            yield result

        elif r.name == "3PC":
            assert r.coeffs.changepoints is not None
            x, y, y_pred = _get_array_right(
                np.array(r.input_data.X),
                np.array(r.input_data.y),
                np.array(r.pred_y),
                r.coeffs.changepoints[0],
            )

            t = r.coeffs.slopes[0] / _std_error(x, y, y_pred)

            if abs(t) > 2.0:
                yield result

        elif r.name == "3PH":
            assert r.coeffs.changepoints is not None
            x, y, y_pred = _get_array_left(
                np.array(r.input_data.X),
                np.array(r.input_data.y),
                np.array(r.pred_y),
                r.coeffs.changepoints[0],
            )

            t = r.coeffs.slopes[0] / _std_error(x, y, y_pred)

            if abs(t) > 2.0:
                yield result

        elif r.name == "4P":
            assert r.coeffs.changepoints is not None
            xl, yl, y_predl = _get_array_left(
                np.array(r.input_data.X),
                np.array(r.input_data.y),
                np.array(r.pred_y),
                r.coeffs.changepoints[0],
            )

            xr, yr, y_predr = _get_array_right(
                np.array(r.input_data.X),
                np.array(r.input_data.y),
                np.array(r.pred_y),
                r.coeffs.changepoints[0],
            )

            tl = r.coeffs.slopes[0] / _std_error(xl, yl, y_predl)
            tr = r.coeffs.slopes[1] / _std_error(xr, yr, y_predr)

            if abs(tl) > 2.0 and abs(tr) > 2.0:
                yield result

        elif r.name == "5P":
            assert r.coeffs.changepoints is not None
            xl, yl, y_predl = _get_array_left(
                np.array(r.input_data.X),
                np.array(r.input_data.y),
                np.array(r.pred_y),
                r.coeffs.changepoints[0],
            )

            xr, yr, y_predr = _get_array_right(
                np.array(r.input_data.X),
                np.array(r.input_data.y),
                np.array(r.pred_y),
                r.coeffs.changepoints[1],
            )

            tl = r.coeffs.slopes[0] / _std_error(xl, yl, y_predl)
            tr = r.coeffs.slopes[1] / _std_error(xr, yr, y_predr)

            if abs(tl) > 2.0 and abs(tr) > 2.0:
                yield result
