from changepointmodel.app import models
import pytest
import pydantic


def test_changepoint_model_request(baseline_request):
    req = models.EnergyChangepointModelRequest(**baseline_request)


def test_changepoint_model_request_with_validation_error_on_models(baseline_request):
    with pytest.raises(pydantic.ValidationError):
        req = baseline_request
        req["model_config"]["models"] = ["33"]
        models.EnergyChangepointModelRequest(**req)


def test_changepoint_model_request_with_validation_error_on_usage(baseline_request):
    req = baseline_request

    with pytest.raises(pydantic.ValidationError):
        req["usage"]["usage"] = [1, 1]
        models.EnergyChangepointModelRequest(**req)

    with pytest.raises(pydantic.ValidationError):
        req["usage"]["usage"] = []
        models.EnergyChangepointModelRequest(**req)


def test_changepoint_model_request_with_validation_error_on_oat(baseline_request):
    with pytest.raises(pydantic.ValidationError):
        req = baseline_request
        req["usage"]["oat"] = []
        models.EnergyChangepointModelRequest(**req)


def test_changepoint_model_request_with_validation_error_for_threshold(
    baseline_request,
):
    req = baseline_request
    req["usage"]["usage"] = [1] * 12
    req["usage"]["oat"] = [1] * 12
    models.EnergyChangepointModelRequest(**req)

    with pytest.raises(pydantic.ValidationError):
        req = baseline_request
        req["usage"]["usage"] = [0] * 10 + [1, 1]
        models.EnergyChangepointModelRequest(**req)

    with pytest.raises(pydantic.ValidationError):
        req = baseline_request
        req["usage"]["oat"] = [0] * 10 + [1, 1]
        models.EnergyChangepointModelRequest(**req)

    req = baseline_request
    req["nonzero_threshold"] = 0.1
    req["usage"]["usage"] = [0] * 10 + [1, 1]  # 2 non-zeros, 2/12 = 0.16666 > 0.1
    models.EnergyChangepointModelRequest(**req)


def test_baseline_model_request_with_norms(baseline_request):
    req = baseline_request
    m = models.BaselineChangepointModelRequest(**req)
    assert m.norms is None

    req["norms"] = [1] * 12
    m = models.BaselineChangepointModelRequest(**req)
    assert m.norms != None

    req["norms"] = [1] * 13
    with pytest.raises(pydantic.ValidationError):
        models.BaselineChangepointModelRequest(**req)


def test_savings_request_no_error(option_c_request):
    req = option_c_request
    m = models.SavingsRequest(**req)


def test_savings_request_with_empty_norms(option_c_request):
    req = option_c_request
    req["norms"] = [1, 2]  # fake here

    with pytest.raises(pydantic.ValidationError):
        models.SavingsRequest(**req)

    req["norms"] = [1] * 12  # fake here
    m = models.SavingsRequest(**req)
    req["norms"] = []  # empty is fine
    m = models.SavingsRequest(**req)


def test_usage_model_with_null():
    oat = [1, None, 1]
    usage = [1, None, 1]

    with pytest.raises(pydantic.ValidationError):
        models.UsageTemperatureData(oat=oat, usage=usage)
