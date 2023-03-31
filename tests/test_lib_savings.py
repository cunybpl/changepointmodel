from changepointmodel.core.calc.savings import adjusted, weather_normalized
from numpy.testing import assert_almost_equal

# XXX these are basically just checking that the math is hitting...
# formulas were copied from bema implementation


def test_adjusted_returns_correct_value():
    res = adjusted(42, 43, 0.5, 2, 12, 12, 0.8)
    # XXX if pre and post are identical we get div by zero error and nans
    # (-1, -0.08333333333333333, -44, 11.321001530179318)
    assert_almost_equal(res, (-1, -0.0833, -0.02, 11.321), decimal=2)


def test_weather_normalized_returns_correct_value():
    res = weather_normalized(100, 101, 0.5, 0.3, 12, 12, 2, 2, 12, 0.8)
    assert_almost_equal(res, (-1, -0.0833, -0.01, 31.517), decimal=2)
    # (-1, -0.08333333333333333, -0.01, 31.517902029409587)
