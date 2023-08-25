import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from changepointmodel.core.nptypes import OneDimNDArray

HeatingCoolingPoints = Tuple[int, int]


def twop(X: OneDimNDArray[np.float64], slope: float) -> HeatingCoolingPoints:
    return (len(X), 0) if slope <= 0 else (0, len(X))


def threepc(X: OneDimNDArray[np.float64], changepoint: float) -> HeatingCoolingPoints:
    return 0, int(sum(X >= changepoint))


def threeph(X: OneDimNDArray[np.float64], changepoint: float) -> HeatingCoolingPoints:
    return int(sum(X <= changepoint)), 0


def fourp(X: OneDimNDArray[np.float64], changepoint: float) -> HeatingCoolingPoints:
    heatnum = sum(X <= changepoint)
    coolnum = sum(X > changepoint)
    return int(heatnum), int(coolnum)


def fivep(
    X: OneDimNDArray[np.float64],
    lcp: float,
    rcp: float,
) -> HeatingCoolingPoints:
    heatnum = sum(X <= lcp)
    coolnum = sum(X >= rcp)
    return int(heatnum), int(coolnum)
