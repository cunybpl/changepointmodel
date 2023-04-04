from changepointmodel.app import extras
from changepointmodel.app.models import FilterHowEnum, FilterWhichEnum


def test_extras_filter_shape_good_models(
    dummy_result_for_filter_extra_good_for_shape_test,
):
    results = dummy_result_for_filter_extra_good_for_shape_test
    filtered = [i for i in extras.shape(results)]
    assert len(filtered) == 4

    for i in filtered:
        assert i.result.name != "5P"


def test_extras_filter_shape_bad_models(
    dummy_result_for_filter_extra_bad_for_shape_test,
):
    results = dummy_result_for_filter_extra_bad_for_shape_test

    for i in extras.shape(results):
        assert i.result.name == "2P"


def test_extras_filter_dpop_good_models(
    dummy_result_for_filter_extra_good_for_dpop_test,
):
    results = dummy_result_for_filter_extra_good_for_dpop_test
    filtered = [i for i in extras.dpop(results)]
    assert len(filtered) == 5


def test_extras_filter_dpop_bad_models(dummy_result_for_filter_extra_bad_for_dpop_test):
    results = dummy_result_for_filter_extra_bad_for_dpop_test
    filtered = [i for i in extras.dpop(results)]
    assert len(filtered) == 1


def test_extras_filter_t_test_good_models(dummy_t_test_good):
    results = dummy_t_test_good
    filtered = [i for i in extras.tstat(results)]
    assert len(filtered) == 5


def test_extras_filter_t_test_bad_models(dummy_t_test_bad):
    results = dummy_t_test_bad
    filtered = [i for i in extras.tstat(results)]
    assert len(filtered) == 1
