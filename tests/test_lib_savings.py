from changepointmodel.core.calc.savings import adjusted, weather_normalized
from numpy.testing import assert_almost_equal

# XXX these are basically just checking that the math is hitting...
# formulas were copied from bema implementation


def test_adjusted_returns_correct_value():
    res = adjusted(42, 43, 0.5, 2, 12, 12, 0.8)
    # XXX if pre and post are identical we get div by zero error and nans

    total_savings, average_savings, percent_savings, percent_savings_uncertainty = res
    assert total_savings == -1
    assert_almost_equal(average_savings, -0.0833, decimal=4)
    assert_almost_equal(percent_savings, -0.0238, decimal=4)
    assert_almost_equal(percent_savings_uncertainty, 11.321001, decimal=4)


def test_weather_normalized_returns_correct_value():
    res = weather_normalized(100, 101, 0.5, 0.3, 12, 12, 2, 2, 12, 0.8)
    total_savings, average_savings, percent_savings, percent_savings_uncertainty = res
    assert total_savings == -1
    assert_almost_equal(average_savings, -0.08333333333333333, decimal=4)
    assert percent_savings == -0.01
    assert_almost_equal(percent_savings_uncertainty, 31.51790, decimal=4)
