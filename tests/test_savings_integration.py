from changepointmodel.core import pmodels as models, estimator
from changepointmodel.core.schemas import CurvefitEstimatorDataModel
import numpy as np
from changepointmodel.core import savings
import pytest


def test_savings_integration_with_pre_and_post(
    generated_3pc_data, generated_4p_post_data
):
    Xdata = np.array(generated_3pc_data["x"])
    ydata = np.array(generated_3pc_data["y"])
    Xdata_post = np.array(generated_4p_post_data["x"])
    ydata_post = np.array(generated_4p_post_data["y"])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    input_data_post = CurvefitEstimatorDataModel(X=Xdata_post, y=ydata_post)

    threepc = models.threepc_factory("3PC")
    estimator_pre = estimator.EnergyChangepointEstimator(model=threepc)

    fourp = models.fourp_factory("4P")
    estimator_post = estimator.EnergyChangepointEstimator(model=fourp)

    X_pre, y_pre, o_pre = input_data.sorted_X_y()
    fitted_est_pre = estimator_pre.fit(X_pre, y_pre)
    X_post, y_post, o_post = input_data_post.sorted_X_y()
    fitted_est_post = estimator_post.fit(X_post, y_post)

    adjusted_saving = savings.AshraeAdjustedSavingsCalculator()
    adj_result = adjusted_saving.save(fitted_est_pre, fitted_est_post)

    norms = np.array([i + 1 for i in Xdata])
    normalized_saving = savings.AshraeNormalizedSavingsCalculator(norms.reshape(-1, 1))
    norm_result = normalized_saving.save(fitted_est_pre, fitted_est_post)

    # adujusted savings
    assert adj_result.total_savings
    assert adj_result.average_savings
    assert adj_result.percent_savings
    assert adj_result.percent_savings_uncertainty

    # normalized savings
    assert norm_result.total_savings
    assert norm_result.average_savings
    assert norm_result.percent_savings
    assert norm_result.percent_savings_uncertainty


def test_savings_integration_with_pre_and_post_savings_check(option_c_request):
    Xdata = np.array(option_c_request["pre"]["usage"]["oat"])
    ydata = np.array(option_c_request["pre"]["usage"]["usage"])
    Xdata_post = np.array(option_c_request["post"]["usage"]["oat"])
    ydata_post = np.array(option_c_request["post"]["usage"]["usage"])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    input_data_post = CurvefitEstimatorDataModel(X=Xdata_post, y=ydata_post)

    # note cannot use fit_one method here for estimator cause
    threepc = models.threepc_factory("3PC")
    estimator_pre = estimator.EnergyChangepointEstimator(model=threepc)

    fourp = models.fourp_factory("4P")
    estimator_post = estimator.EnergyChangepointEstimator(model=fourp)

    X_pre, y_pre, o_pre = input_data.sorted_X_y()
    fitted_est_pre = estimator_pre.fit(X_pre, y_pre)
    X_post, y_post, o_post = input_data_post.sorted_X_y()
    fitted_est_post = estimator_post.fit(X_post, y_post)

    adjusted_saving = savings.AshraeAdjustedSavingsCalculator()
    adj_result = adjusted_saving.save(fitted_est_pre, fitted_est_post)

    norms = np.array([i + 1 for i in Xdata])
    normalized_saving = savings.AshraeNormalizedSavingsCalculator(norms.reshape(-1, 1))
    norm_result = normalized_saving.save(fitted_est_pre, fitted_est_post)

    # saving check
    assert pytest.approx(adj_result.total_savings, 1e-4) == 3221.2695
    assert pytest.approx(norm_result.total_savings, 1e-4) == 2764.7594
