import numpy as np
from changepointmodel.core.calc import bounds


def test_twop_bounds():
    res = bounds.twop()
    assert res == ((0, -np.inf), (np.inf, np.inf))


def test_threepc_bounds():
    check_X = np.linspace(1.0, 10.0)
    res = bounds.threepc(check_X)
    assert res == ((0, 0, 3.2040816326530615), (np.inf, np.inf, 7.795918367346939))


def test_threeph_bounds():
    check_X = np.linspace(1.0, 10.0)
    res = bounds.threeph(check_X)
    assert res == ((0, -np.inf, 3.2040816326530615), (np.inf, 0, 7.795918367346939))


def test_fourp_bounds():
    check_X = np.linspace(1.0, 10.0)
    res = bounds.fourp(check_X)
    assert res == (
        (0, -np.inf, -np.inf, 3.2040816326530615),
        (np.inf, np.inf, np.inf, 7.795918367346939),
    )


def test_fivep_bounds():
    check_X = np.linspace(1.0, 10.0)
    res = bounds.fivep(check_X)
    assert res == (
        (0, -np.inf, 0, 3.2040816326530615, 6.6938775510204085),
        (np.inf, 0, np.inf, 4.3061224489795915, 7.795918367346939),
    )
