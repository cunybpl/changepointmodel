from changepointmodel.core.predsum import PredictedSum, PredictedSumCalculator
from changepointmodel.core.estimator import EnergyChangepointEstimator

import numpy as np


def test_predicted_sum(mocker):
    est = EnergyChangepointEstimator()

    check = np.array(
        [
            [
                1.0,
            ],
            [
                2.0,
            ],
            [
                3.0,
            ],
            [
                4.0,
            ],
        ]
    )
    mocker.patch.object(
        est, "predict", return_value=check
    )  # just return the check from predict and see if its summed

    calc = PredictedSumCalculator(check)
    res = calc.calculate(est)
    assert res.value == 10
