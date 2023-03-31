from changepointmodel.bema import filter_
from changepointmodel.bema.models import FilterHowEnum, FilterWhichEnum


def test_filter_best_score_with_good_model(dummy_result_for_filter_good):
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.best_score,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(dummy_result_for_filter_good)
    assert len(results) == 1
    best_result = results.pop()
    assert best_result.result.name == 3


def test_filter_best_score_with_bad_model(dummy_result_for_filter_bad):
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.best_score,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(dummy_result_for_filter_bad)
    assert len(results) == 1
    best_result = results.pop()
    assert best_result.result.name == 5


def test_filter_best_score_with_mixed_models(
    dummy_result_for_filter_bad, dummy_result_for_filter_good
):
    contianers = dummy_result_for_filter_bad + dummy_result_for_filter_good

    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.best_score,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(contianers)
    assert len(results) == 1
    best_result = results.pop()
    assert best_result.result.name == 3


def test_filter_threshold_ok_with_good_model(dummy_result_for_filter_good):
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.threshold_ok,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(dummy_result_for_filter_good)
    assert len(results) == 3


def test_filter_threshold_ok_with_bad_model(dummy_result_for_filter_bad):
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.threshold_ok,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(dummy_result_for_filter_bad)
    assert len(results) == 0  # return zero models for bad models =; no fall back


def test_filter_threshold_ok_with_mixed_model(
    dummy_result_for_filter_bad, dummy_result_for_filter_good
):
    containers = dummy_result_for_filter_bad + dummy_result_for_filter_good
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.threshold_ok,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(containers)
    assert len(results) == 3


def test_filter_threshold_ok_first_is_best_with_good_model(
    dummy_result_for_filter_good,
):
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.threshold_ok_first_is_best,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(dummy_result_for_filter_good)
    assert len(results) == 1
    best_result = results.pop()
    assert best_result.result.name == 1


def test_filter_threshold_ok_first_is_best_with_bad_model(dummy_result_for_filter_bad):
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.threshold_ok_first_is_best,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(dummy_result_for_filter_bad)
    assert len(results) == 0  # return zero models for bad models =; no fall back


def test_filter_threshold_ok_first_is_best_with_mixed_model(
    dummy_result_for_filter_bad, dummy_result_for_filter_good
):
    containers = dummy_result_for_filter_bad + dummy_result_for_filter_good
    filter_dict = {
        "which": FilterWhichEnum.r2,
        "how": FilterHowEnum.threshold_ok_first_is_best,
        "extras": False,
    }
    filter_obj = filter_.ChangepointEstimatorFilter(**filter_dict)
    results = filter_obj.filtered(containers)
    assert len(results) == 1
    best_result = results.pop()
    assert best_result.result.name == 1
