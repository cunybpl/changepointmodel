import pytest


def test_imports_bema():
    from changepointmodel.app import AppChangepointModeler, SavingsResponse


def test_imports_changepoint_core():
    from changepointmodel.core import (
        EnergyChangepointEstimator,
        EnergyChangepointLoadsAggregator,
        CurvefitEstimator,
        CurvefitEstimatorDataModel,
    )


def test_imports_changepoint():
    from changepointmodel.app import AppChangepointModeler, SavingsResponse
    from changepointmodel.core import (
        EnergyChangepointEstimator,
        EnergyChangepointLoadsAggregator,
        CurvefitEstimator,
        CurvefitEstimatorDataModel,
    )
