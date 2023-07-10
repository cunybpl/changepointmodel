from changepointmodel.core import utils
from changepointmodel.core.pmodels import (
    EnergyParameterModelCoefficients,
    FourParameterModel,
    ParameterModelFunction,
)
import numpy as np
from changepointmodel.core.pmodels.coeffs_parser import FourParameterCoefficientsParser


def test_argsort_1d_idx():
    x = np.array(
        [
            [
                5,
            ],
            [
                3,
            ],
            [
                2,
            ],
            [
                1,
            ],
            [
                4,
            ],
        ]
    )  # [3 2 1 4 0 ] is sorting index
    y = np.array([5, 3, 2, 1, 4])

    _x, _y, ordering = utils.argsort_1d_idx(x, y)

    assert _y.tolist() == [1, 2, 3, 4, 5]
    assert ordering.tolist() == [3, 2, 1, 4, 0]


def test_argsort_unargsort_pair():
    x_ = [
        31.5549430847168,
        41.46236038208008,
        40.40647888183594,
        48.85377502441406,
        66.41313934326172,
        72.4892578125,
        79.4146957397461,
        80.5838623046875,
        72.34444427490234,
        59.53846740722656,
        45.7861213684082,
        40.545326232910156,
    ]

    y_ = [
        2771.7328796075267,
        2778.9696134985625,
        2866.628408060809,
        2887.0038965747117,
        3553.987575239154,
        3980.2125375555543,
        4595.253582350758,
        4622.158449907303,
        4136.844967333332,
        3386.2120536070365,
        2841.454855450504,
        2862.490733376347,
    ]
    x = np.array(x_)
    y = np.array(y_)

    new_x, new_y, ordering = utils.argsort_1d_idx(x, y)

    sorted_x = [
        i.pop()
        for i in [
            [31.5549430847168],
            [40.40647888183594],
            [40.545326232910156],
            [41.46236038208008],
            [45.7861213684082],
            [48.85377502441406],
            [59.53846740722656],
            [66.41313934326172],
            [72.34444427490234],
            [72.4892578125],
            [79.4146957397461],
            [80.5838623046875],
        ]
    ]

    sorted_y = [y_[i] for i in ordering]

    assert list(new_x) == sorted_x
    assert list(new_y) == sorted_y

    unorder_x = utils.unargsort_1d_idx(new_x, ordering)
    unorder_y = utils.unargsort_1d_idx(new_y, ordering)

    assert list(unorder_x) == x_
    assert list(unorder_y) == y_


def test_parse_coeffs():
    m = ParameterModelFunction(
        "dumb", None, None, FourParameterModel(), FourParameterCoefficientsParser()
    )
    res = utils.parse_coeffs(m, (1, 2, 3, 4))
    assert isinstance(res, EnergyParameterModelCoefficients)
