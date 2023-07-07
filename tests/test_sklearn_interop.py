import pytest
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from changepointmodel.core.pmodels import (
    ParameterModelFunction,
    AbstractEnergyParameterModel,
    ICoefficientParser,
)
from changepointmodel.core.estimator import EnergyChangepointEstimator
from sklearn.utils.validation import check_is_fitted

from numpy.testing import assert_array_almost_equal, assert_array_equal


def test_estimator_works_with_cross_val_functions():
    # XXX this test also implies that lower level cross_validate is operable
    # which can operate on multiple metrics. Potential to cross val cp models on both cvrmse and r2 but
    # will require some custom setup and a bit of experimentation
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    class LineModel(AbstractEnergyParameterModel):
        pass

    class LineCoeffParse(ICoefficientParser):
        pass

    bounds = ((0, -np.inf), (np.inf, np.inf))
    mymodel = ParameterModelFunction(
        "line", line, bounds, LineModel(), LineCoeffParse()
    )

    X = np.linspace(1, 100, 100).reshape(-1, 1)
    y = np.linspace(1, 100, 100)
    est = EnergyChangepointEstimator(model=mymodel)

    scores = cross_val_score(
        est,
        X,
        y,
        cv=3,
    )

    assert_array_almost_equal(scores, [1.0, 1.0, 1.0])

    # remember this does not fit/predict only score
    # https://stackoverflow.com/questions/42263915/using-sklearn-cross-val-score-and-kfolds-to-fit-and-help-predict-model
    # 1. use this to crossvalidate an average score (over training data)
    # 2. determine if this result is good enough and then fit (over training data)
    # 3. test against your holdout (unseen) data.

    predicted = cross_val_predict(est, X, y, cv=3)
    assert_array_almost_equal(y, predicted, decimal=1)


def test_estimator_works_with_gridsearchcv():
    def line(X, yint, m):
        return (m * X + yint).squeeze()

    def curve(X, a, b, c):
        return (a * np.exp(-b * X) + c).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    curve_bounds = ((0, 0, 0), (10, 10, 10))

    X = np.linspace(1, 100, 100).reshape(-1, 1)
    y = np.linspace(1, 100, 100)

    line_model = ParameterModelFunction("line", line, bounds)
    curve_model = ParameterModelFunction("curve", curve, curve_bounds)

    grid = {"model": [line_model, curve_model]}

    est = EnergyChangepointEstimator()
    search = GridSearchCV(est, param_grid=grid, cv=3)
    search.fit(X, y)

    assert (
        search.best_estimator_.name == "line"
    )  # if this picks a curve over a line then all is lost...
    assert_array_almost_equal(search.predict(X), y, decimal=1)
