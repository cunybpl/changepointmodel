from changepointmodel.core.pmodels import (
    FiveParameterCoefficientsParser,
    FiveParameterModel,
    FourParameterCoefficientsParser,
    FourParameterModel,
    ParameterModelFunction,
    ParameterModelFunction,
    ThreeParameterCoefficientsParser,
    ThreeParameterCoolingModel,
    ThreeParameterHeatingModel,
    TwoParameterCoefficientParser,
    TwoParameterModel,
    YinterceptMixin,
    EnergyParameterModelCoefficients,
)
import numpy as np


def test_yinterceptmixin():
    coeffs = EnergyParameterModelCoefficients(42, [99], [])

    model = YinterceptMixin()
    assert model.yint(coeffs) == 42


def test_twoparametermodel():
    coeffs = EnergyParameterModelCoefficients(42, [99], [])

    model = TwoParameterModel()
    assert model.slope(coeffs) == 99


def test_threeparametercoolingmodel():
    coeffs = EnergyParameterModelCoefficients(42, [99], [43])
    model = ThreeParameterCoolingModel()
    assert model.slope(coeffs) == 99
    assert model.changepoint(coeffs) == 43


def test_threeparameterheatingmodel():
    coeffs = EnergyParameterModelCoefficients(42, [99], [43])
    model = ThreeParameterHeatingModel()
    assert model.slope(coeffs) == 99
    assert model.changepoint(coeffs) == 43


def test_fourparametermodel():
    coeffs = EnergyParameterModelCoefficients(42, [98, 99], [43])
    model = FourParameterModel()
    assert model.right_slope(coeffs) == 99
    assert model.left_slope(coeffs) == 98
    assert model.changepoint(coeffs) == 43


def test_fiveparametermodel():
    coeffs = EnergyParameterModelCoefficients(42, [98, 99], [43, 44])
    model = FiveParameterModel()
    assert model.right_slope(coeffs) == 99
    assert model.left_slope(coeffs) == 98
    assert model.right_changepoint(coeffs) == 44
    assert model.left_changepoint(coeffs) == 43


def test_linear_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [99], [])
    parser = TwoParameterCoefficientParser()
    assert parser.parse((42, 99)) == test


def test_singleslope_slinglechangepoint_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [99], [43])
    parser = ThreeParameterCoefficientsParser()
    assert parser.parse((42, 99, 43)) == test


def test_dualslope_singlechangepoint_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [98, 99], [43])
    parser = FourParameterCoefficientsParser()
    assert parser.parse((42, 98, 99, 43)) == test


def test_dualslope_dualchangepoint_coefficient_parser():
    test = EnergyParameterModelCoefficients(42, [98, 99], [43, 44])
    parser = FiveParameterCoefficientsParser()
    assert parser.parse((42, 98, 99, 43, 44)) == test


def test_modelfunction():
    def f(X, y):
        return (X + y).squeeze()

    bound = (42,), (43,)
    parmeter_model = TwoParameterModel()
    parser = TwoParameterCoefficientParser()

    model = ParameterModelFunction(
        "mymodel",
        f=f,
        bounds=bound,
        parameter_model=parmeter_model,
        coefficients_parser=parser,
    )

    assert model.name == "mymodel"
    assert model.f == f
    assert model.bounds == bound
    assert model.parameter_model == parmeter_model

    assert model.parse_coeffs((42, 99)) == EnergyParameterModelCoefficients(
        42, [99], []
    )
