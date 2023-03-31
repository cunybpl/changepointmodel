import numpy as np
from changepointmodel.core.calc import loads


def test_baseload_calculation():
    assert 0 == loads.baseload(100, 50, 50)


def test_postive_sum_only_sums_positives():
    assert loads._positive_sum(np.array([-1.0, 0.0, 1.0])) == 1


def test_coolingload_calculates_correct_predicted_load():
    test_X = np.linspace(1, 10, 10)
    test_y = np.linspace(1, 10, 10)
    yint = 2.0
    changepoint = 5.0

    res = loads._cooling_predicted_load(test_X, test_y, yint, changepoint)
    assert [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ] == res.tolist()


def test_heatingload_calculates_correct_predicted_load():
    test_X = np.linspace(1, 10, 10)
    test_y = np.linspace(10, 1, 10)
    yint = 2.0
    changepoint = 5.0

    res = loads._heating_predicted_load(test_X, test_y, yint, changepoint)
    assert [8.0, 7.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] == res.tolist()


def test_coolingload_returns_correct_result(mocker):
    test_X = np.linspace(1, 10, 10)
    test_y = np.linspace(1, 10, 10)
    yint = 2.0
    changepoint = 5.0

    coolspy = mocker.spy(loads, "_cooling_predicted_load")
    sumspy = mocker.spy(loads, "_positive_sum")

    l = loads.coolingload(test_X, test_y, yint, changepoint)
    assert l == 4 + 5 + 6 + 7 + 8

    coolspy.assert_called_once_with(test_X, test_y, yint, changepoint)

    calls = sumspy.call_args_list
    np.testing.assert_array_equal(
        calls[0][0][0],
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ],
    )  # XXX prob should make a helper function out of this


def test_heatingload_correct_calling_behavior(mocker):
    test_X = np.linspace(1, 10, 10)
    test_y = np.linspace(10, 1, 10)
    yint = 2.0
    changepoint = 5.0

    heatspy = mocker.spy(loads, "_heating_predicted_load")
    sumspy = mocker.spy(loads, "_positive_sum")

    l = loads.heatload(test_X, test_y, yint, changepoint)
    assert l == 8 + 7 + 6 + 5
    heatspy.assert_called_once_with(test_X, test_y, yint, changepoint)

    calls = sumspy.call_args_list
    np.testing.assert_array_equal(
        calls[0][0][0], [8.0, 7.0, 6.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
