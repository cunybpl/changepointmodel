from changepointmodel.core.calc import uncertainties
from numpy.testing import assert_almost_equal


def test_fractional_avoided_energy_use():
    res = uncertainties.fractional_avoided_energy_use(100, 100)
    assert res == 0


def test_relative_uncertainty_avoided_energy_use():
    res = uncertainties.relative_uncertainty_avoided_energy_use(0.8, 0.5, 1, 2, 12, 12)
    assert_almost_equal(res, 0.269, decimal=3)


def test_relatieve_uncertainity_normalized_period():
    res = uncertainties.relative_uncertainty_normalized_period(0.8, 12, 0.5, 100, 2, 12)
    assert_almost_equal(res, 26.954, decimal=3)
