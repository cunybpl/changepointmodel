from changepointmodel.app import base, exc, main, models

import pytest


def test_AppChangepointModeler(raw_energy_model_data):
    usage = raw_energy_model_data["pre_data"]["usage"]
    oat = raw_energy_model_data["pre_data"]["oat"]

    models = ["2P", "3PC", "3PH", "4P"]

    cpm = main.ChangepointModelerApplication(oat=oat, usage=usage, models=models)

    results = cpm.run()
    assert len(results) == 4
    res = results.pop(0)
    assert isinstance(res, base.ChangepointResultContainer)

    res_pred_y = res.result.pred_y
    res_input_data_x = res.result.input_data.X
    res_input_data_y = res.result.input_data.y

    ex_pred_y = [i * res.result.coeffs.slopes[0] + res.result.coeffs.yint for i in oat]
    assert list(res_input_data_x) == oat
    assert list(res_input_data_y) == usage
    assert list(res_pred_y) == ex_pred_y


def test_AppChangepointModeler_with_exception(raw_energy_model_data):
    usage = raw_energy_model_data["pre_data"]["usage"]
    oat = raw_energy_model_data["pre_data"]["oat"]

    models = ["5P"]

    cpm = main.ChangepointModelerApplication(oat=oat, usage=usage, models=models)

    with pytest.raises(exc.ChangepointException):
        cpm.run()


def test_run_baseline(baseline_request):
    request = models.BaselineChangepointModelRequest(**baseline_request)
    response = main.run_baseline(request)

    assert len(response.results) == 1

    # wihtout filter
    req = baseline_request
    req["model_config"].pop("model_filter")
    request = models.BaselineChangepointModelRequest(**req)
    response = main.run_baseline(request)


def test_run_baseline_with_cvrmse_threshold(baseline_request):
    req = baseline_request
    req["model_config"]["model_filter"]["which"] = "cvrmse"
    req["model_config"]["model_filter"]["how"] = "best_score"
    request = models.BaselineChangepointModelRequest(**req)
    response = main.run_baseline(request)

    assert len(response.results) == 1


def test_run_baseline_with_extras(baseline_request):
    req = baseline_request
    req["model_config"]["model_filter"]["extras"] = True
    request = models.BaselineChangepointModelRequest(**req)
    response = main.run_baseline(request)

    assert len(response.results) == 1


def test_run_baseline_with_norms(baseline_request):
    request = models.BaselineChangepointModelRequest(**baseline_request)
    response = main.run_baseline(request)

    assert len(response.results) == 1

    # wihtout filter
    req = baseline_request
    req["norms"] = [1] * 12  # adding norms to see if it runs
    req["model_config"].pop("model_filter")
    request = models.BaselineChangepointModelRequest(**req)
    response = main.run_baseline(request)

    assert len(response.results) == 4


def test_run_baseline_with_exception(baseline_request):
    req = baseline_request
    req["model_config"]["models"].append("5P")
    request = models.BaselineChangepointModelRequest(**req)

    with pytest.raises(exc.ChangepointException):
        main.run_baseline(request)


def test_run_option_c_with_filter(option_c_request):
    req = option_c_request
    req["pre"]["model_config"]["models"] = ["2P", "3PC", "3PH", "4P"]
    request = models.SavingsRequest(**req)
    response = main.run_optionc(request)

    assert len(response.results) == 1  # only one best model for pre and post


def test_run_option_c_with_filter_with_norms(option_c_request):
    req = option_c_request
    req["pre"]["model_config"]["models"] = ["2P", "3PC", "3PH", "4P"]
    req["norms"] = [1] * 12
    request = models.SavingsRequest(**req)
    response = main.run_optionc(request)

    assert len(response.results) == 1  # only one best model for pre and post
    res = response.results[0]

    assert len(res.normalized_savings.result.normalized_y_post) == 12
    assert len(res.normalized_savings.result.normalized_y_pre) == 12


def test_run_option_c_with_no_filter(option_c_request, raw_energy_model_data):
    req = option_c_request
    req["pre"]["model_config"]["models"] = ["2P", "3PC", "3PH", "4P"]
    req["pre"]["model_config"].pop("model_filter")
    req["post"]["model_config"].pop("model_filter")
    request = models.SavingsRequest(**req)
    response = main.run_optionc(request)

    assert len(response.results) == 20  # 4 models for pre and 5 models for post

    res = response.results.pop(0)

    res_pred_y = res.pre.pred_y
    res_input_data_x = res.pre.input_data.X
    res_input_data_y = res.pre.input_data.y

    usage = raw_energy_model_data["pre_data"]["usage"]
    oat = raw_energy_model_data["pre_data"]["oat"]

    post_oat = raw_energy_model_data["post_data"]["oat"]

    cp_model = lambda x: x * res.pre.coeffs.slopes[0] + res.pre.coeffs.yint
    ex_pred_y = [cp_model(i) for i in oat]
    assert list(res_input_data_x) == oat
    assert list(res_input_data_y) == usage
    assert list(res_pred_y) == ex_pred_y

    assert res.pre.name == "2P"
    assert res.post.name == "2P"

    exp_adj_y = [cp_model(i) for i in post_oat]
    res_adj_y = res.adjusted_savings.result.adjusted_y

    assert list(res_adj_y) == exp_adj_y


def test_run_option_c_with_exception(option_c_request):
    req = option_c_request
    request = models.SavingsRequest(**req)

    with pytest.raises(exc.ChangepointException):
        main.run_optionc(request)


def test_run_option_c_with_exception_in_post(option_c_request):
    req = {
        "pre": option_c_request["post"],
        "post": option_c_request["pre"],
        "confidence_interval": 0.8,
        "scalar": None,
    }
    request = models.SavingsRequest(**req)

    with pytest.raises(exc.ChangepointException):
        main.run_optionc(request)
