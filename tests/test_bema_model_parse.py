from changepointmodel.bema import models
import pytest
import json


def test_EnergyChangepointModelResult_parse(changepointmodel_response_fixture):
    data = changepointmodel_response_fixture["results"][0]
    result = models.EnergyChangepointModelResult(**data)

    # the model is 2P
    flat_cp_result = result.assemble_full_cp_model_result()

    measurements = [
        "base_load",
        "heating_load",
        "cooling_load",
        "cooling_sensitivity",
        "heating_sensitivity",
        "heating_changepoint",
        "cooling_changepoint",
        "normalized_annual_consumption",
        "r2",
        "r2_threshold",
        "r2_ok",
        "cvrmse",
        "cvrmse_threshold",
        "cvrmse_ok",
        "model_type",
    ]

    for i in measurements:
        assert i in flat_cp_result.keys()

    assert flat_cp_result["model_type"] == "2P"

    none_keys = [
        "heating_sensitivity",
        "heating_changepoint",
        "cooling_changepoint",
        "normalized_annual_consumption",
    ]
    for i in none_keys:
        assert flat_cp_result[i] == None

    assert flat_cp_result["heating_load"] == 0.0

    not_none_keys = [
        "r2",
        "r2_threshold",
        "r2_ok",
        "cvrmse",
        "cvrmse_threshold",
        "cvrmse_ok",
        "base_load",
        "cooling_load",
    ]

    for i in not_none_keys:
        assert flat_cp_result[i]

    # changed nac
    data["nac"] = {"value": 600}
    result = models.EnergyChangepointModelResult(**data)
    flat_cp_result = result.assemble_full_cp_model_result()
    assert flat_cp_result["normalized_annual_consumption"] == 600


def test_EnergyChangepointModelResult_parse_for_db(changepointmodel_response_fixture):
    data = changepointmodel_response_fixture["results"][0]
    result = models.EnergyChangepointModelResult(**data)

    cpmodel_result = result.make_cp_model_result()
    assert "r2" in cpmodel_result
    assert "cvrmse" in cpmodel_result
    assert "result" in cpmodel_result

    assert cpmodel_result["r2"] > 0
    assert cpmodel_result["cvrmse"] > 0

    assert "input_data" in cpmodel_result["result"]["cpmodel"]
    assert "oat" in cpmodel_result["result"]["cpmodel"]["input_data"]
    assert "usage" in cpmodel_result["result"]["cpmodel"]["input_data"]

    assert len(cpmodel_result["result"]["cpmodel"]["input_data"]["oat"]) != 0
    assert len(cpmodel_result["result"]["cpmodel"]["input_data"]["usage"]) != 0

    jsonstr = json.dumps(cpmodel_result)
    json_dict = json.loads(jsonstr)  # load correctly


def test_SavingsResult_parse(savings_fixture):
    data = savings_fixture["results"][0]

    result = models.SavingsResult(**data)

    # test cp parsing
    cp_models = result.parse_prepost_cp_model_results()
    assert len(cp_models) == 2

    act_prepost = set([i["prepost"] for i in cp_models])
    exp_prespost = {"pre", "post"}
    assert act_prepost == exp_prespost

    # test savings results parsing; example: average_savings, percent_savings etc.
    saving_results = result.parse_savings_result()

    saving_types = [i["savings_type"] for i in saving_results]
    assert len(saving_types) == 2
    assert set(saving_types) == {"adjusted", "normalized"}
    saving_result_measure_types = {
        "average_savings",
        "percent_savings",
        "percent_savings_uncertainty",
        "total_savings",
    }

    for i in saving_results:
        keys = set(i.keys())
        assert saving_result_measure_types.issubset(keys)  # all keys present

    # test saving usage; for example, adjusted saving usage, normalized saving usage etc.

    adjusted_savings_usage, normalized_savings_usage = result.parse_saving_usage()

    pre_adjusted_usage = [i for i in adjusted_savings_usage if i["prepost"] == "pre"]
    post_adjusted_usage = [i for i in adjusted_savings_usage if i["prepost"] == "post"]

    assert len(pre_adjusted_usage) == len(post_adjusted_usage)

    # adjusted usage for pre is always none; only add this for completeness of schemas
    for i in pre_adjusted_usage:
        assert i["adjusted_usage"] is None

    for i in post_adjusted_usage:
        assert i["adjusted_usage"] != None

    pre_norm_usage = [i for i in normalized_savings_usage if i["prepost"] == "pre"]
    post_norm_usage = [i for i in normalized_savings_usage if i["prepost"] == "post"]

    assert len(pre_norm_usage) == len(post_norm_usage)
    assert len(pre_norm_usage) == 12  # always twelve months

    pre_months = [i["month"] for i in pre_norm_usage]
    post_months = [i["month"] for i in post_norm_usage]

    # check months
    for index_, month in enumerate(list(range(1, 13))):
        assert month == pre_months[index_]
        assert month == post_months[index_]


def test_SavingsResult_parse_json_test(savings_fixture):
    data = savings_fixture["results"][0]

    result = models.SavingsResult(**data)

    # test cp parsing
    cp_models = result.parse_prepost_cp_model_results()
    jsonstr = json.dumps(cp_models)
    json_dict = json.loads(jsonstr)


def test_EnergyChangepointModelResponse_parsing(changepointmodel_response_fixture):
    data = changepointmodel_response_fixture
    result = models.EnergyChangepointModelResponse(**data)

    parsed_model, predicted_usage = result.parse_results_for_csv()
    assert len(parsed_model) == 1  # only one model here

    for i in predicted_usage:
        assert i["prepost"] == "pre"
        assert i["predicted_usage"] != None

    # XXX see the functions for comments; just testing it here for completeness
    parsed_measurements = result.parse_cp_measurements()
    for i in parsed_measurements:
        assert "model_type" in i


def test_SavingsResponse_parsing(savings_fixture):
    data = savings_fixture
    result = models.SavingsResponse(**data)

    # testing public method
    (
        model_output,
        savings_results_output,
        savings_related_usage_output,
        norms_output,
    ) = result.parse_results_for_csv()


# yint: float
#     slopes: List[float]
#     changepoints: Optional[List[float]]
def test_EnergyParameterModelCoefficients_parsing_2P():
    model_type = "2P"

    # cooling
    data = {"yint": 1, "slopes": [1], "changepoints": None}

    load_result = models.EnergyParameterModelCoefficients(**data)

    parsed = load_result.parse(model_type)
    assert len(parsed) == 1  # only one measurementtype for 2P; sensitivity

    assert parsed[0]["measurementtype"] == "cooling_sensitivity"
    assert parsed[0]["value"] == 1

    data = {"yint": 1, "slopes": [-1], "changepoints": None}

    load_result = models.EnergyParameterModelCoefficients(**data)

    parsed = load_result.parse(model_type)
    assert len(parsed) == 1  # only one measurementtype for 2P; sensitivity

    assert parsed[0]["measurementtype"] == "heating_sensitivity"
    assert parsed[0]["value"] == -1


def test_EnergyParameterModelCoefficients_parsing_assertion_error():
    model_type = "dum"

    # cooling
    data = {"yint": 1, "slopes": [1], "changepoints": None}

    load_result = models.EnergyParameterModelCoefficients(**data)
    with pytest.raises(AssertionError):
        load_result.parse(model_type)


def test_EnergyParameterModelCoefficients_parsing_3PC():
    model_type = "3PC"

    # cooling
    data = {"yint": 1, "slopes": [1], "changepoints": [1]}

    load_result = models.EnergyParameterModelCoefficients(**data)

    parsed = load_result.parse(model_type)
    assert (
        len(parsed) == 2
    )  # two measurementtypes for 3PC; cooling sensitivity and cooling changepoint

    assert parsed[0]["measurementtype"] == "cooling_sensitivity"
    assert parsed[0]["value"] == 1

    assert parsed[1]["measurementtype"] == "cooling_changepoint"
    assert parsed[1]["value"] == 1

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [1, 2], "changepoints": [1]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [1], "changepoints": [1, 2]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)


def test_EnergyParameterModelCoefficients_parsing_3PH():
    model_type = "3PH"

    # heating
    data = {"yint": 1, "slopes": [-1], "changepoints": [1]}

    load_result = models.EnergyParameterModelCoefficients(**data)

    parsed = load_result.parse(model_type)
    assert (
        len(parsed) == 2
    )  # two measurementtypes for 3PH; heating sensitivity and heating changepoint

    assert parsed[0]["measurementtype"] == "heating_sensitivity"
    assert parsed[0]["value"] == -1

    assert parsed[1]["measurementtype"] == "heating_changepoint"
    assert parsed[1]["value"] == 1

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [1, 2], "changepoints": [1]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [1], "changepoints": [1, 2]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)


def test_EnergyParameterModelCoefficients_parsing_4P():
    model_type = "4P"

    # heating
    data = {"yint": 1, "slopes": [-1, 1], "changepoints": [1]}

    load_result = models.EnergyParameterModelCoefficients(**data)

    parsed = load_result.parse(model_type)
    assert (
        len(parsed) == 4
    )  # four measurementtypes for 4P; sensitivity and changepoint both cooling and heating; but same changepoint

    assert parsed[0]["measurementtype"] == "heating_sensitivity"
    assert parsed[0]["value"] == -1

    assert parsed[1]["measurementtype"] == "cooling_sensitivity"
    assert parsed[1]["value"] == 1

    assert parsed[2]["measurementtype"] == "heating_changepoint"
    assert parsed[2]["value"] == 1

    assert parsed[3]["measurementtype"] == "cooling_changepoint"
    assert parsed[3]["value"] == 1

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [1], "changepoints": [1]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [-1, 1], "changepoints": [1, 2]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)


def test_EnergyParameterModelCoefficients_parsing_5P():
    model_type = "5P"

    # heating
    data = {"yint": 1, "slopes": [-1, 1], "changepoints": [1, 2]}

    load_result = models.EnergyParameterModelCoefficients(**data)

    parsed = load_result.parse(model_type)
    assert (
        len(parsed) == 4
    )  # four measurementtypes for 5P; sensitivity and changepoint both cooling and heating; not the same changepoint

    assert parsed[0]["measurementtype"] == "heating_sensitivity"
    assert parsed[0]["value"] == -1

    assert parsed[1]["measurementtype"] == "cooling_sensitivity"
    assert parsed[1]["value"] == 1

    assert parsed[2]["measurementtype"] == "heating_changepoint"
    assert parsed[2]["value"] == 1

    assert parsed[3]["measurementtype"] == "cooling_changepoint"
    assert parsed[3]["value"] == 2

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [1], "changepoints": [1, 2]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)

    with pytest.raises(AssertionError):
        data = {"yint": 1, "slopes": [-1, 1], "changepoints": [1]}
        load_result = models.EnergyParameterModelCoefficients(**data)
        parsed = load_result.parse(model_type)
