import pytest


def test_imports_bema():
    from changepointmodel.bema import BemaChangepointModeler, SavingsResponse


def test_imports_changepoint_core():
    from changepointmodel.core import (
        EnergyChangepointEstimator,
        EnergyChangepointLoadsAggregator,
        CurvefitEstimator,
        CurvefitEstimatorDataModel,
    )


def test_imports_changepoint():
    from changepointmodel.bema import BemaChangepointModeler, SavingsResponse
    from changepointmodel.core import (
        EnergyChangepointEstimator,
        EnergyChangepointLoadsAggregator,
        CurvefitEstimator,
        CurvefitEstimatorDataModel,
    )
