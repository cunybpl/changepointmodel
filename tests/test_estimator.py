from sklearn.exceptions import NotFittedError
from changepointmodel.core.pmodels import ModelFunction
from changepointmodel.core.estimator import CurvefitEstimator
import numpy as np

from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)
from changepointmodel.core.estimator import EnergyChangepointEstimator
import pytest

from numpy.testing import assert_array_almost_equal


def test_energychangepointestimator_fit_calls_curvefitestimator_fit(mocker):
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ModelFunction("line", line, bounds)

    est = EnergyChangepointEstimator(model=mymodel)
    mock = mocker.spy(est, "fit")

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    est.fit(X, y)
    mock.assert_called_once()

    assert_almost_equal(est.predict(X), y, decimal=1)


def test_energychangepointestimator_properties_set_on_fit():
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ModelFunction("line", line, bounds)

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    est = EnergyChangepointEstimator(mymodel)
    est.fit(X, y)

    assert est.name == "line"

    assert_array_equal(X, est.X)
    assert_array_equal(y, est.y)
    assert_almost_equal((0, 1), est.coeffs, decimal=1)
    assert_array_almost_equal(est.pred_y, y, decimal=1)
    assert est.sigma is None
    assert est.cov.all()
    assert est.absolute_sigma is False


def test_estimator_calculated_getter_methods():
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ModelFunction("line", line, bounds)

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    est = EnergyChangepointEstimator(mymodel)
    est.fit(X, y)

    assert est.n_params() == 2
    assert round(est.total_pred_y()) == np.sum(y)
    assert est.total_y() == np.sum(y)
    assert est.len_y() == len(y)


def test_unfit_estimator_raises_notfittederror_on_property_access():
    est = EnergyChangepointEstimator()

    with pytest.raises(NotFittedError):
        est.X

    with pytest.raises(NotFittedError):
        est.y

    with pytest.raises(NotFittedError):
        est.pred_y

    with pytest.raises(NotFittedError):
        est.sigma

    with pytest.raises(NotFittedError):
        est.absolute_sigma

    with pytest.raises(NotFittedError):
        est.cov


def test_unfit_estimator_raises_notfitted_error_on_method_calls():
    est = EnergyChangepointEstimator()
    X = 42

    with pytest.raises(NotFittedError):
        est.predict(X)

    with pytest.raises(NotFittedError):
        est.predict(X)

    with pytest.raises(NotFittedError):
        est.adjust(X)


def test_name_accessor_raises_valueerror_if_model_not_set():
    est = EnergyChangepointEstimator()
    with pytest.raises(ValueError):
        est.name


def test_estimator_fit_one():
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ModelFunction("line", line, bounds)

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    name, est = EnergyChangepointEstimator.fit_one(mymodel, X, y)
    assert name == "line"
    assert est.pred_y.all()  # assert that we have predictions


def test_estimator_fit_many():
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    models = [ModelFunction(f"{i}", line, bounds) for i in range(5)]

    counter = 0
    for name, est in EnergyChangepointEstimator.fit_many(models, X, y):
        assert int(name) == counter
        assert est.pred_y.all()
        counter += 1


def test_estimator_fit_one_fail_silently(mocker):
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ModelFunction("line", line, bounds)

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    mocker.patch.object(CurvefitEstimator, "fit", side_effect=Exception("boo"))

    est, name = EnergyChangepointEstimator.fit_one(mymodel, X, y)
    assert est == "line"
    assert name is None

    with pytest.raises(Exception):
        EnergyChangepointEstimator.fit_one(mymodel, X, y, fail_silently=False)


def test_estimator_fit_many_fail_silently(mocker):
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    models = [ModelFunction(f"{i}", line, bounds) for i in range(5)]

    mocker.patch.object(CurvefitEstimator, "fit", side_effect=Exception("boo"))

    counter = 0
    for name, est in EnergyChangepointEstimator.fit_many(models, X, y):
        assert int(name) == counter
        assert est is None
        counter += 1


def test_estimator_adjust_calls_predict_with_other_x(mocker):
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ModelFunction("line", line, bounds)

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    X1 = np.linspace(10, 1, 10).reshape(-1, 1)
    y1 = np.linspace(10, 1, 10)

    est1 = EnergyChangepointEstimator(mymodel)
    est1.fit(X, y)

    est2 = EnergyChangepointEstimator(mymodel)
    est2.fit(X1, y1)

    mock = mocker.spy(est1, "predict")
    est1.adjust(est2)
    assert_array_equal(mock.call_args_list[0][0][0], X1)
