import pytest


def test_imports_app():
    from changepointmodel.app import ChangepointModelerApplication, SavingsResponse


def test_imports_changepoint_core():
    from changepointmodel.core import (
        EnergyChangepointEstimator,
        EnergyChangepointLoadsAggregator,
        CurvefitEstimator,
        CurvefitEstimatorDataModel,
    )


def test_imports_changepoint():
    from changepointmodel.app import ChangepointModelerApplication, SavingsResponse
    from changepointmodel.core import (
        EnergyChangepointEstimator,
        EnergyChangepointLoadsAggregator,
        CurvefitEstimator,
        CurvefitEstimatorDataModel,
    )
