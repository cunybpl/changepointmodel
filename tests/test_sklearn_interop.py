# # XXX temporarily removed for 3.1
# import pytest
# import numpy as np
# from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
# from changepointmodel.core.pmodels import (
#     ParameterModelFunction,
#     AbstractEnergyParameterModel,
#     ICoefficientParser,
# )
# from changepointmodel.core.estimator import CurvefitEstimator
# from sklearn.utils.validation import check_is_fitted

# from numpy.testing import assert_array_almost_equal, assert_array_equal


# def test_estimator_works_with_cross_val_functions():
#     # XXX this test also implies that lower level cross_validate is operable
#     # which can operate on multiple metrics. Potential to cross val cp models on both cvrmse and r2 but
#     # will require some custom setup and a bit of experimentation
#     def line(X, yint, m):
#         return (m * X + yint).squeeze()

#     bounds = ((0, -np.inf), (np.inf, np.inf))
#     # mymodel = ParameterModelFunction(
#     #     "line", line, bounds
#     # )

#     X = np.linspace(1, 100, 100).reshape(-1, 1)
#     y = np.linspace(1, 100, 100)
#     est = CurvefitEstimator(model_func=line, bounds=bounds, lsq_kwargs={})

#     scores = cross_val_score(
#         est,
#         X,
#         y,
#         cv=3,
#     )

#     assert_array_almost_equal(scores, [1.0, 1.0, 1.0])

#     #     # remember this does not fit/predict only score
#     #     # https://stackoverflow.com/questions/42263915/using-sklearn-cross-val-score-and-kfolds-to-fit-and-help-predict-model
#     #     # 1. use this to crossvalidate an average score (over training data)
#     #     # 2. determine if this result is good enough and then fit (over training data)
#     #     # 3. test against your holdout (unseen) data.

#     predicted = cross_val_predict(est, X, y, cv=3)
#     assert_array_almost_equal(y, predicted, decimal=1)


# # def test_estimator_works_with_gridsearchcv():
# #     def line(X, yint, m):
# #         return (m * X + yint).squeeze()

# #     def curve(X, a, b, c):
# #         return (a * np.exp(-b * X) + c).squeeze()

# #     bounds = ((0, -np.inf), (np.inf, np.inf))
# #     curve_bounds = ((0, 0, 0), (10, 10, 10))

# #     X = np.linspace(1, 100, 100).reshape(-1, 1)
# #     y = np.linspace(1, 100, 100)

# #     # line_model = ParameterModelFunction("line", line, bounds)
# #     # curve_model = ParameterModelFunction("curve", curve, curve_bounds)

# #     grid = {"model": [line, curve]}

# #     est = CurvefitEstimator(model_func=line, bounds=bounds)
# #     search = GridSearchCV(est, param_grid=grid, cv=3)
# #     search.fit(X, y)

# #     assert (
# #         search.best_estimator_.name_ == "line"
# #     )  # if this picks a curve over a line then all is lost...
# #     assert_array_almost_equal(search.predict(X), y, decimal=1)


import numpy as np
from changepointmodel.core.estimator import CurvefitEstimator
from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


np.random.seed(1729)


def test_curvefit_estimator_against_scipy_example():
    # Use the scipy example to fit a function.

    def func(x, a, b, c):
        return (a * np.exp(-b * x) + c).squeeze()

    bounds = ((0, -np.inf), (np.inf, np.inf))
    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    estimator = CurvefitEstimator(model_func=func)
    estimator.fit(xdata.reshape(-1, 1), ydata)
    expected = [2.5542, 1.3519, 0.4745]
    assert list(expected) == [round(p, 4) for p in list(estimator.popt_)]
    estimator.predict(xdata.reshape(-1, 1))  # more or less just a smoke test


def test_curvefit_estimator_with_pipeline_api():
    # smoke test to make sure we are compatible with sklearn pipeline API
    def func(x, a, b, c):
        return (a * np.exp(-b * x) + c).squeeze()

    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    estimator = CurvefitEstimator(model_func=func)
    estimators = [("ct", StandardScaler()), ("f", estimator)]

    pipe = pipeline.Pipeline(estimators)
    pipe.fit(xdata.reshape(-1, 1), ydata)
    pipe.predict(xdata.reshape(-1, 1))


def test_curvefit_grid_search():
    # smoketest that estimator works in a grid search

    def f(x, a, b, c):
        return (a * np.exp(-b * x) + c).squeeze()

    def g(x, a, b, c, d):
        return (a * np.exp(b * x) + c - d).squeeze()

    xdata = np.linspace(0, 4, 50)
    y = f(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    params = {
        "model_func": [f, g],
    }

    search = GridSearchCV(CurvefitEstimator(), param_grid=params)
    search.fit(xdata.reshape(-1, 1), ydata)
    assert search.best_estimator_.name_ == "f"


def test_callable_bounds():
    def f(x, a, b, c):
        return (a * np.exp(-b * x) + c).squeeze()

    xdata = np.linspace(0, 4, 50)
    y = f(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    def bounds(X):
        return (-np.inf, np.inf)

    est = CurvefitEstimator(model_func=f, bounds=bounds)
    est.fit(xdata.reshape(-1, 1), ydata)


def test_X_and_y_are_accessible_on_estimator_after_fit():
    def f(x, a, b, c):
        return (a * np.exp(-b * x) + c).squeeze()

    xdata = np.linspace(0, 4, 50)
    y = f(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    est = CurvefitEstimator(model_func=f)
    est.fit(xdata.reshape(-1, 1), ydata)
    assert hasattr(est, "X_")
    assert hasattr(est, "y_")
    assert est.X_.all() == xdata.all()
    assert est.y_.all() == ydata.all()
