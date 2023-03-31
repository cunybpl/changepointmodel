from changepointmodel.core import scoring
import numpy as np


def test_r2_forwards_call(mocker):
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    mock = mocker.patch("changepointmodel.core.calc.metrics.r2_score")

    method = scoring.R2()
    method(y, y_pred)
    mock.assert_called_once()


def test_rmse_forwards_call(mocker):
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    mock = mocker.patch("changepointmodel.core.calc.metrics.rmse")

    method = scoring.Rmse()
    method(y, y_pred)
    mock.assert_called_once()


def test_cvrmse_forwards_call(mocker):
    y = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([4.0, 5.0, 6.0])
    mock = mocker.patch("changepointmodel.core.calc.metrics.cvrmse")

    method = scoring.Cvrmse()
    method(y, y_pred)
    mock.assert_called_once()


def test_scoreeval(score_mock_estimator, score_mock_scorefunction):
    seval = scoring.ScoreEval(score_mock_scorefunction, 42.0, lambda a, b: a == b)
    res = seval.ok(score_mock_estimator)

    assert res.name == "dummy"
    assert res.threshold == 42.0
    assert res.ok == True
    assert res.value == 42.0


def test_scorer(score_mock_estimator):
    class DummyScoreEval(scoring.IEval):
        def ok(self, estimator):
            return scoring.Score(name="dumb", value=42.0, threshold=42.0, ok=True)

    evals = [DummyScoreEval(), DummyScoreEval(), DummyScoreEval()]

    s = scoring.Scorer(evals)
    res = s.check(score_mock_estimator)

    for r in res:
        assert r.name == "dumb"
        assert r.threshold == 42.0
        assert r.ok == True
        assert r.value == 42.0
