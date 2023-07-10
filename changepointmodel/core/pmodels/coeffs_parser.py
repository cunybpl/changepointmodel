from typing import Tuple
from .base import ICoefficientParser, EnergyParameterModelCoefficients


class TwoParameterCoefficientParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, slope = coeffs
        return EnergyParameterModelCoefficients(yint, [slope], [])


class ThreeParameterCoefficientsParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, slope, changepoint = coeffs
        return EnergyParameterModelCoefficients(yint, [slope], [changepoint])


class FourParameterCoefficientsParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, ls, rs, changepoint = coeffs
        return EnergyParameterModelCoefficients(yint, [ls, rs], [changepoint])


class FiveParameterCoefficientsParser(ICoefficientParser):
    def parse(self, coeffs: Tuple[float, ...]) -> EnergyParameterModelCoefficients:
        yint, ls, rs, lcp, rcp = coeffs
        return EnergyParameterModelCoefficients(yint, [ls, rs], [lcp, rcp])
