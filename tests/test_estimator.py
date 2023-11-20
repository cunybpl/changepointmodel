from sklearn.exceptions import NotFittedError
from changepointmodel.core.pmodels import (
    ParameterModelFunction,
    TwoParameterModel,
)
from changepointmodel.core.pmodels.coeffs_parser import TwoParameterCoefficientParser

from changepointmodel.core.estimator import CurvefitEstimator
from changepointmodel.core.calc.models import twop
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
    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ParameterModelFunction(
        "2P", twop, bounds, TwoParameterModel(), TwoParameterCoefficientParser()
    )

    est = EnergyChangepointEstimator(model=mymodel)
    mock = mocker.spy(est, "fit")

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    est.fit(X, y)
    mock.assert_called_once()

    assert_almost_equal(est.predict(X), y, decimal=1)


def test_energychangepointestimator_properties_set_on_fit():
    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ParameterModelFunction(
        "2P", twop, bounds, TwoParameterModel(), TwoParameterCoefficientParser()
    )

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    est = EnergyChangepointEstimator(mymodel)
    est.fit(X, y)

    assert est.name == "2P"

    assert_array_equal(X, est.X)
    assert_array_equal(y, est.y)
    assert_almost_equal((0, 1), est.coeffs, decimal=1)
    assert_array_almost_equal(est.pred_y, y, decimal=1)
    assert est.sigma is None
    assert est.cov.all()
    assert est.absolute_sigma is False


def test_estimator_calculated_getter_methods():
    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ParameterModelFunction(
        "2P", twop, bounds, TwoParameterModel(), TwoParameterCoefficientParser()
    )

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


def test_estimator_adjust_calls_predict_with_other_x(mocker):
    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ParameterModelFunction(
        "2P", twop, bounds, TwoParameterModel(), TwoParameterCoefficientParser()
    )

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


def test_estimator_for_methods(mocker):
    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ParameterModelFunction(
        "2P", twop, bounds, TwoParameterModel(), TwoParameterCoefficientParser()
    )

    est = EnergyChangepointEstimator(model=mymodel)
    mock = mocker.spy(est, "fit")

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    est.fit(X, y)
    mock.assert_called_once()

    assert_almost_equal(est.predict(X), y, decimal=1)
    assert_almost_equal(est.r2(), 1, decimal=1)
    assert_almost_equal(est.adjusted_r2(), 1, decimal=1)
    assert_almost_equal(est.rmse(), 2.32678e-05, decimal=1)
    assert_almost_equal(est.cvrmse(), 4.23051e-06, decimal=1)
    assert est.dpop() == (0, 10)
    assert_almost_equal(est.tstat()[1], 349150.818, decimal=1)
    assert est.shape() == True
    load_ = est.load()
    assert_almost_equal(load_.cooling, 54.999605, decimal=1)
    assert_almost_equal(load_.base, sum(X) - load_.cooling, decimal=1)


def test_estimator_scalar_handling_for_load_and_nac(mocker):
    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ParameterModelFunction(
        "2P", twop, bounds, TwoParameterModel(), TwoParameterCoefficientParser()
    )

    est = EnergyChangepointEstimator(model=mymodel)
    mock = mocker.spy(est, "fit")

    X = np.linspace(1, 10, 10).reshape(-1, 1)
    y = np.linspace(1, 10, 10)

    est.fit(X, y)
    mock.assert_called_once()

    load_not_scaled = est.load()
    load_scaled = est.load(scalar=30.437)

    assert_almost_equal(load_scaled.base, load_not_scaled.base * 30.437)
    assert_almost_equal(load_scaled.cooling, load_not_scaled.cooling * 30.437)
    assert_almost_equal(load_scaled.heating, load_not_scaled.heating * 30.437)

    nac_not_scaled = est.nac(X)
    nac_scaled = est.nac(X, scalar=30.437)

    assert_almost_equal(nac_scaled, nac_not_scaled * 30.437)
