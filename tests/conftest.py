from changepointmodel.core.pmodels import EnergyParameterModelCoefficients
from changepointmodel.core import (
    estimator,
    scoring,
    schemas,
    savings,
    loads,
    predsum,
    utils,
)
import pytest
import numpy as np
import json
from typing import Optional, List, Union
from dataclasses import dataclass
from . import GENERATED_DATA_ALL_MODELS_FILE, GENERATED_DATA_POST, ENERGYMODEL_DATAPATH


@pytest.fixture
def schema_scores():
    return [
        {"name": "r2", "value": 0, "threshold": 0, "ok": False},
        {"name": "cvrmse", "value": 0, "threshold": 0, "ok": False},
    ]


@pytest.fixture
def schema_load():
    return {"base": 0, "cooling": 0, "heating": 0}


@pytest.fixture
def schema_coeffs():
    return {"yint": 42.0, "slopes": [1.0], "changepoints": None}


@pytest.fixture
def schema_adjustedsavings():
    return {
        "confidence_interval": 0.8,
        "result": {
            "adjusted_y": np.array([1.0, 2.0, 3.0]),
            "total_savings": 42.0,
            "average_savings": 42.0,
            "percent_savings": 42.0,
            "percent_savings_uncertainty": 42.0,
        },
    }


@pytest.fixture
def schema_normalizedsavings():
    return {
        "X_pre": np.array(
            [
                [
                    1.0,
                ]
            ]
        ),
        "X_post": np.array(
            [
                [
                    1.0,
                ]
            ]
        ),
        "confidence_interval": 0.8,
        "result": {
            "normalized_y_pre": np.array([1.0, 2.0, 3.0]),
            "normalized_y_post": np.array([1.0, 2.0, 3.0]),
            "total_savings": 42.0,
            "average_savings": 42.0,
            "percent_savings": 42.0,
            "percent_savings_uncertainty": 42.0,
        },
    }


@pytest.fixture
def score_mock_estimator():
    class ScoreMockEstimator(estimator.EnergyChangepointEstimator):
        y_ = np.array([1.0, 2.0, 3.0])
        pred_y_ = np.array([4.0, 5.0, 6.0])

    return ScoreMockEstimator()


@pytest.fixture
def score_mock_scorefunction():
    class Dummy(scoring.ScoringFunction):
        def calc(self, y, pred_y, **kwargs):
            return 42.0

    return Dummy()


@pytest.fixture
def dummy_twopcoefficients():
    return EnergyParameterModelCoefficients(98, [99], None)


@pytest.fixture
def dummy_threepcoefficients():
    return EnergyParameterModelCoefficients(98, [99], 100)


@pytest.fixture
def dummy_fourpcoefficients():
    return EnergyParameterModelCoefficients(98, [99, 100], 101)


@pytest.fixture
def dummy_fivepcoefficients():
    return EnergyParameterModelCoefficients(98, [99, 100], [101, 102])


@pytest.fixture
def loads_dummyestimator(loads_dummyenergyparametermodel):
    # this is tricky to mock...
    class _estimator(object):
        popt_ = (
            99,
            99,
        )

    class LoadsDummyEstimator(estimator.EnergyChangepointEstimator):
        estimator_ = _estimator
        X_ = np.array(
            [
                [
                    1.0,
                ],
            ]
        )
        pred_y_ = np.array(
            [
                1.0,
            ]
        )

    return LoadsDummyEstimator(model=loads_dummyenergyparametermodel)


@pytest.fixture
def generated_data_all_models():
    with open(GENERATED_DATA_ALL_MODELS_FILE, "r") as f:
        return json.load(f)


@pytest.fixture
def generated_data_for_post():
    with open(GENERATED_DATA_POST, "r") as f:
        return json.load(f)


def _parse_generated_mode_data(data, model_type):
    for i in data:
        if i["model_type"] == model_type:
            return i


@pytest.fixture
def generated_2p_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, "2P")


@pytest.fixture
def generated_3pc_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, "3PC")


@pytest.fixture
def generated_3ph_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, "3PH")


@pytest.fixture
def generated_4p_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, "4P")


@pytest.fixture
def generated_5p_data(generated_data_all_models):
    return _parse_generated_mode_data(generated_data_all_models, "5P")


@pytest.fixture
def generated_2p_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, "2P")


@pytest.fixture
def generated_3pc_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, "3PC")


@pytest.fixture
def generated_3ph_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, "3PH")


@pytest.fixture
def generated_4p_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, "4P")


@pytest.fixture
def generated_5p_post_data(generated_data_for_post):
    return _parse_generated_mode_data(generated_data_for_post, "5P")


class EnergyChangepointModelResultFactory(object):
    @classmethod
    def create(
        cls,
        estimator: estimator.EnergyChangepointEstimator,
        loads: Optional[loads.EnergyChangepointLoadsAggregator] = None,
        scorer: Optional[scoring.Scorer] = None,
        nac: Optional[predsum.PredictedSumCalculator] = None,
    ):
        """Constructs a EnergyChangepointModelResult given at least a fit estimator. Optionally will
        provide scores, nac and loads if given configured instances of their handlers.

        Args:
            estimator (EnergyChangepointEstimator): energy changpoint estimator object.
            loads (Optional[loads.EnergyChangepointLoadsAggregator], optional): load object. Defaults to None.
            scorer (Optional[scoring.Scorer], optional): scorer. Defaults to None.
            nac (Optional[PredictedSumCalculator], optional): nac. Defaults to None.

        Returns:
            schemas.EnergyChangepointModelResult: [description]
        """

        input_data = schemas.CurvefitEstimatorDataModel(
            X=estimator.X,
            y=estimator.y,
            sigma=estimator.sigma,
            absolute_sigma=estimator.absolute_sigma,
        )

        data = {
            "name": estimator.name,
            "input_data": {
                "X": estimator.X,
                "y": estimator.y,
                "sigma": estimator.sigma,
                "absolute_sigma": estimator.absolute_sigma,
            },
            "coeffs": utils.parse_coeffs(estimator.model, estimator.coeffs),
            "pred_y": estimator.pred_y,
            "load": loads.aggregate(estimator) if loads else None,
            "scores": scorer.check(estimator) if scorer else None,
            "nac": nac.calculate(estimator) if nac else None,
        }

        return data


class SavingsResultFactory(object):
    @classmethod
    def create(
        cls,
        pre: estimator.EnergyChangepointEstimator,
        post: estimator.EnergyChangepointEstimator,
        adjcalc: savings.AshraeAdjustedSavingsCalculator,
        normcalc: Optional[savings.AshraeNormalizedSavingsCalculator] = None,
        pre_loads: Optional[loads.EnergyChangepointLoadsAggregator] = None,
        post_loads: Optional[loads.EnergyChangepointLoadsAggregator] = None,
        pre_scorer: Optional[scoring.Scorer] = None,
        post_scorer: Optional[scoring.Scorer] = None,
    ):
        """Creates a SavingsResult given pre and post retrofit models. Designed for usage with the
        option-c methodology.

        Args:
            pre (EnergyChangepointEstimator): Pre changepoint estimator obj
            post (EnergyChangepointEstimator): Post changepoint estimator obj
            adjcalc (AshraeAdjustedSavingsCalculator): Adjusted calc instance
            normcalc (Optional[AshraeNormalizedSavingsCalculator], optional): normalized calc instance. Defaults to None.
            pre_loads (Optional[loads.EnergyChangepointLoadsAggregator], optional): Pre load obj. Defaults to None.
            post_loads (Optional[loads.EnergyChangepointLoadsAggregator], optional): Post load obj. Defaults to None.
            pre_scorer (Optional[Scorer], optional): scorer for pre modeling. Defaults to None.
            post_scorer (Optional[Scorer], optional): scorer for post modeling. Defaults to None.

        Returns:
            schemas.SavingsResult: The SavingsResult.
        """

        pre_result = EnergyChangepointModelResultFactory.create(
            pre, pre_loads, pre_scorer
        )
        post_result = EnergyChangepointModelResultFactory.create(
            post, post_loads, post_scorer
        )

        adj = dict(
            result=adjcalc.save(pre, post),
            confidence_interval=adjcalc.confidence_interval,
        )

        if normcalc:
            result = normcalc.save(pre, post)
            norm = dict(
                X_pre=normcalc.X_norms,
                X_post=normcalc.X_norms,
                confidence_interval=normcalc.confidence_interval,
                result=result,
            )
        else:
            norm = None

        data = {
            "pre": pre_result,
            "post": post_result,
            "adjusted_savings": adj,
            "normalized_savings": norm,
        }
        return data


# ports from bplrpc for changepoint.bema pacakge


@pytest.fixture
def raw_energy_model_data():
    with open(ENERGYMODEL_DATAPATH, "r") as f:
        return json.load(f)


@pytest.fixture
def baseline_request(raw_energy_model_data):
    usage = raw_energy_model_data["pre_data"]
    models = ["2P", "3PC", "3PH", "4P"]
    model_filter = {"which": "r2", "how": "threshold_ok_first_is_best"}
    return {
        "model_config": dict(models=models, model_filter=model_filter),
        "usage": usage,
    }


@pytest.fixture
def option_c_request(raw_energy_model_data):
    models = ["2P", "3PC", "3PH", "4P", "5P"]
    norms = [i + 1 for i in raw_energy_model_data["pre_data"]["oat"]]
    model_filter = {"which": "r2", "how": "best_score"}
    pre = dict(
        model_config=dict(models=models, norms=norms, model_filter=model_filter),
        usage=raw_energy_model_data["pre_data"],
    )
    post = dict(
        model_config=dict(models=models, norms=norms, model_filter=model_filter),
        usage=raw_energy_model_data["post_data"],
    )
    return dict(pre=pre, post=post, confidence_interval=0.8, scalar=None)


@dataclass
class DummyScore:
    name: str
    value: float
    threshold: float
    ok: bool


@dataclass
class DummyCoefficients:
    yint: float
    slopes: List[float]
    changepoints: Optional[List[float]]


@dataclass
class DummyInputData:
    X: np.array
    y: np.array


@dataclass
class DummResult:
    scores: Optional[
        List[DummyScore]
    ] = None  # making it optional so that testing is eaiser for filter extras
    coeffs: Optional[
        DummyCoefficients
    ] = None  # making it optional so that test dont break
    pred_y: Optional[List[float]] = None  # making it optional so that test dont break
    input_data: Optional[
        DummyInputData
    ] = None  # making it optional so that test dont break
    name: Union[str, int] = None  # added later to track the mdoel


@dataclass
class DummyResultContainer:
    result: DummResult


@pytest.fixture
def dummy_result_for_filter_good():
    score_1_1 = DummyScore(name="r2", value=0.8, threshold=0.75, ok=True)
    score_2_1 = DummyScore(name="r2", value=0.85, threshold=0.75, ok=True)
    score_3_1 = DummyScore(name="r2", value=0.9, threshold=0.75, ok=True)

    result_1 = DummResult(scores=[score_1_1], name=1)
    result_2 = DummResult(scores=[score_2_1], name=2)
    result_3 = DummResult(scores=[score_3_1], name=3)

    dummy_container_1 = DummyResultContainer(result=result_1)
    dummy_container_2 = DummyResultContainer(result=result_2)
    dummy_container_3 = DummyResultContainer(result=result_3)

    return [dummy_container_1, dummy_container_2, dummy_container_3]


@pytest.fixture
def dummy_result_for_filter_bad():
    score_1_1 = DummyScore(name="r2", value=0.1, threshold=0.75, ok=False)
    score_2_1 = DummyScore(name="r2", value=0.2, threshold=0.75, ok=False)

    result_1 = DummResult(scores=[score_1_1], name=4)
    result_2 = DummResult(scores=[score_2_1], name=5)

    dummy_container_1 = DummyResultContainer(result=result_1)
    dummy_container_2 = DummyResultContainer(result=result_2)

    return [dummy_container_1, dummy_container_2]


@pytest.fixture
def dummy_result_for_filter_extra_good_for_shape_test():
    coeffs_2p = DummyCoefficients(yint=1, slopes=[1], changepoints=None)
    coeffs_3pc = DummyCoefficients(yint=1, slopes=[1], changepoints=[5])
    coeffs_3ph = DummyCoefficients(yint=1, slopes=[-1], changepoints=[5])
    coeffs_4p = DummyCoefficients(yint=1, slopes=[-1.5, 1], changepoints=[5])
    coeffs_5p = DummyCoefficients(
        yint=1, slopes=[-0.5, 1], changepoints=[5, 6]
    )  # making this bad intentionally to test slopes comparison

    result_2p = DummResult(coeffs=coeffs_2p, name="2P")
    result_3pc = DummResult(coeffs=coeffs_3pc, name="3PC")
    result_3ph = DummResult(coeffs=coeffs_3ph, name="3PH")
    result_4p = DummResult(coeffs=coeffs_4p, name="4P")
    result_5p = DummResult(coeffs=coeffs_5p, name="5P")

    result_container_2p = DummyResultContainer(result=result_2p)
    result_container_3pc = DummyResultContainer(result=result_3pc)
    result_container_3ph = DummyResultContainer(result=result_3ph)
    result_container_4p = DummyResultContainer(result=result_4p)
    result_container_5p = DummyResultContainer(result=result_5p)

    return [
        result_container_2p,
        result_container_3pc,
        result_container_3ph,
        result_container_4p,
        result_container_5p,
    ]


@pytest.fixture
def dummy_result_for_filter_extra_bad_for_shape_test():
    coeffs_2p = DummyCoefficients(yint=1, slopes=[1], changepoints=None)
    coeffs_3pc = DummyCoefficients(yint=1, slopes=[-1], changepoints=[5])
    coeffs_3ph = DummyCoefficients(yint=1, slopes=[1], changepoints=[5])
    coeffs_4p = DummyCoefficients(yint=1, slopes=[1, -1], changepoints=[5])
    coeffs_5p = DummyCoefficients(yint=1, slopes=[1, -1], changepoints=[5, 6])

    result_2p = DummResult(coeffs=coeffs_2p)
    result_3pc = DummResult(coeffs=coeffs_3pc)
    result_3ph = DummResult(coeffs=coeffs_3ph)
    result_4p = DummResult(coeffs=coeffs_4p)
    result_5p = DummResult(coeffs=coeffs_5p)

    result_container_2p = DummyResultContainer(result=result_2p)
    result_container_3pc = DummyResultContainer(result=result_3pc)
    result_container_3ph = DummyResultContainer(result=result_3ph)
    result_container_4p = DummyResultContainer(result=result_4p)
    result_container_5p = DummyResultContainer(result=result_5p)

    return [
        result_container_2p,
        result_container_3pc,
        result_container_3ph,
        result_container_4p,
        result_container_5p,
    ]


@pytest.fixture
def dummy_result_for_filter_extra_good_for_dpop_test():
    data = DummyInputData(
        X=np.array([i for i in range(1, 13)]), y=np.array([i for i in range(1, 13)])
    )
    coeffs_2p = DummyCoefficients(yint=1, slopes=[1], changepoints=None)
    coeffs_3pc = DummyCoefficients(yint=1, slopes=[1], changepoints=[5])
    coeffs_3ph = DummyCoefficients(yint=1, slopes=[-1], changepoints=[5])
    coeffs_4p = DummyCoefficients(yint=1, slopes=[-1.5, 1], changepoints=[5])
    coeffs_5p = DummyCoefficients(yint=1, slopes=[-0.5, 1], changepoints=[5, 9])

    result_2p = DummResult(coeffs=coeffs_2p, name="2P", input_data=data)
    result_3pc = DummResult(coeffs=coeffs_3pc, name="3PC", input_data=data)
    result_3ph = DummResult(coeffs=coeffs_3ph, name="3PH", input_data=data)
    result_4p = DummResult(coeffs=coeffs_4p, name="4P", input_data=data)
    result_5p = DummResult(coeffs=coeffs_5p, name="5P", input_data=data)

    result_container_2p = DummyResultContainer(result=result_2p)
    result_container_3pc = DummyResultContainer(result=result_3pc)
    result_container_3ph = DummyResultContainer(result=result_3ph)
    result_container_4p = DummyResultContainer(result=result_4p)
    result_container_5p = DummyResultContainer(result=result_5p)

    return [
        result_container_2p,
        result_container_3pc,
        result_container_3ph,
        result_container_4p,
        result_container_5p,
    ]


@pytest.fixture
def dummy_result_for_filter_extra_bad_for_dpop_test():
    data = DummyInputData(
        X=np.array([i for i in range(1, 13)]), y=np.array([i for i in range(1, 13)])
    )
    coeffs_2p = DummyCoefficients(yint=1, slopes=[1], changepoints=None)
    coeffs_3pc = DummyCoefficients(yint=1, slopes=[1], changepoints=[2])
    coeffs_3ph = DummyCoefficients(yint=1, slopes=[-1], changepoints=[2])
    coeffs_4p = DummyCoefficients(yint=1, slopes=[-1.3, 1], changepoints=[2])
    coeffs_5p = DummyCoefficients(yint=1, slopes=[-0.5, 1], changepoints=[5, 6])

    result_2p = DummResult(coeffs=coeffs_2p, name="2P", input_data=data)
    result_3pc = DummResult(coeffs=coeffs_3pc, name="3PC", input_data=data)
    result_3ph = DummResult(coeffs=coeffs_3ph, name="3PH", input_data=data)
    result_4p = DummResult(coeffs=coeffs_4p, name="4P", input_data=data)
    result_5p = DummResult(coeffs=coeffs_5p, name="5P", input_data=data)

    result_container_2p = DummyResultContainer(result=result_2p)
    result_container_3pc = DummyResultContainer(result=result_3pc)
    result_container_3ph = DummyResultContainer(result=result_3ph)
    result_container_4p = DummyResultContainer(result=result_4p)
    result_container_5p = DummyResultContainer(result=result_5p)

    return [
        result_container_2p,
        result_container_3pc,
        result_container_3ph,
        result_container_4p,
        result_container_5p,
    ]


@pytest.fixture
def dummy_t_test_good():
    x = np.array([i for i in range(1, 13)])
    y_2p = np.array([i for i in range(1, 13)])
    data_2p = DummyInputData(X=x, y=y_2p)

    func_3pc = lambda x: x if x >= 5 else 1
    y_3pc = [func_3pc(i) for i in range(1, 13)]
    data_3pc = DummyInputData(X=x, y=np.array(y_3pc))

    func_3ph = lambda x: -x if x <= 5 else 1
    y_3ph = [func_3ph(i) for i in range(1, 13)]
    data_3ph = DummyInputData(X=x, y=np.array(y_3ph))

    func_4p = lambda x: -x if x <= 5 else x
    y_4p = [func_4p(i) for i in range(1, 13)]
    data_4p = DummyInputData(X=x, y=np.array(y_4p))

    func_5p = lambda x: -x if x <= 5 else x if x >= 9 else 5
    y_5p = [func_5p(i) for i in range(1, 13)]
    data_5p = DummyInputData(X=x, y=np.array(y_5p))

    coeffs_2p = DummyCoefficients(yint=1, slopes=[1], changepoints=None)
    coeffs_3pc = DummyCoefficients(yint=1, slopes=[1], changepoints=[5])
    coeffs_3ph = DummyCoefficients(yint=1, slopes=[-1], changepoints=[5])
    coeffs_4p = DummyCoefficients(yint=1, slopes=[-1.5, 1], changepoints=[5])
    coeffs_5p = DummyCoefficients(yint=1, slopes=[-0.5, 1], changepoints=[5, 9])

    result_2p = DummResult(coeffs=coeffs_2p, name="2P", input_data=data_2p, pred_y=y_2p)
    result_3pc = DummResult(
        coeffs=coeffs_3pc, name="3PC", input_data=data_3pc, pred_y=y_3pc
    )
    result_3ph = DummResult(
        coeffs=coeffs_3ph, name="3PH", input_data=data_3ph, pred_y=y_3ph
    )
    result_4p = DummResult(coeffs=coeffs_4p, name="4P", input_data=data_4p, pred_y=y_4p)
    result_5p = DummResult(coeffs=coeffs_5p, name="5P", input_data=data_5p, pred_y=y_5p)

    result_container_2p = DummyResultContainer(result=result_2p)
    result_container_3pc = DummyResultContainer(result=result_3pc)
    result_container_3ph = DummyResultContainer(result=result_3ph)
    result_container_4p = DummyResultContainer(result=result_4p)
    result_container_5p = DummyResultContainer(result=result_5p)

    return [
        result_container_2p,
        result_container_3pc,
        result_container_3ph,
        result_container_4p,
        result_container_5p,
    ]


@pytest.fixture
def dummy_t_test_bad():
    x = np.array([i for i in range(1, 13)])
    y_2p = np.array([i for i in range(1, 13)])
    data_2p = DummyInputData(X=x, y=y_2p)

    func_3pc = lambda x: x if x >= 5 else 1
    y_3pc = [func_3pc(i) for i in range(1, 13)]
    data_3pc = DummyInputData(X=x, y=np.array(y_3pc))
    pred_y_3pc = [i + 5 for i in y_3pc]

    func_3ph = lambda x: -x if x <= 5 else 1
    y_3ph = [func_3ph(i) for i in range(1, 13)]
    data_3ph = DummyInputData(X=x, y=np.array(y_3ph))
    pred_y_3ph = [i + 5 for i in y_3ph]

    func_4p = lambda x: -x if x <= 5 else x
    y_4p = [func_4p(i) for i in range(1, 13)]
    data_4p = DummyInputData(X=x, y=np.array(y_4p))
    pred_y_4p = [i + 5 for i in y_4p]

    func_5p = lambda x: -x if x <= 5 else x if x >= 9 else 5
    y_5p = [func_5p(i) for i in range(1, 13)]
    data_5p = DummyInputData(X=x, y=np.array(y_5p))
    pred_y_5p = [i + 5 for i in y_5p]

    coeffs_2p = DummyCoefficients(yint=1, slopes=[1], changepoints=None)
    coeffs_3pc = DummyCoefficients(yint=1, slopes=[1], changepoints=[5])
    coeffs_3ph = DummyCoefficients(yint=1, slopes=[-1], changepoints=[5])
    coeffs_4p = DummyCoefficients(yint=1, slopes=[-1.5, 1], changepoints=[5])
    coeffs_5p = DummyCoefficients(yint=1, slopes=[-0.5, 1], changepoints=[5, 9])

    result_2p = DummResult(coeffs=coeffs_2p, name="2P", input_data=data_2p, pred_y=y_2p)
    result_3pc = DummResult(
        coeffs=coeffs_3pc, name="3PC", input_data=data_3pc, pred_y=pred_y_3pc
    )
    result_3ph = DummResult(
        coeffs=coeffs_3ph, name="3PH", input_data=data_3ph, pred_y=pred_y_3ph
    )
    result_4p = DummResult(
        coeffs=coeffs_4p, name="4P", input_data=data_4p, pred_y=pred_y_4p
    )
    result_5p = DummResult(
        coeffs=coeffs_5p, name="5P", input_data=data_5p, pred_y=pred_y_5p
    )

    result_container_2p = DummyResultContainer(result=result_2p)
    result_container_3pc = DummyResultContainer(result=result_3pc)
    result_container_3ph = DummyResultContainer(result=result_3ph)
    result_container_4p = DummyResultContainer(result=result_4p)
    result_container_5p = DummyResultContainer(result=result_5p)

    return [
        result_container_2p,
        result_container_3pc,
        result_container_3ph,
        result_container_4p,
        result_container_5p,
    ]


# response for parsing
@pytest.fixture
def changepointmodel_response_fixture():
    return {
        "results": [
            {
                "name": "2P",
                "coeffs": {
                    "yint": 1156.3440318420405,
                    "slopes": [40.34016591248294],
                    "changepoints": None,
                },
                "pred_y": [
                    2429.2756712384726,
                    2786.3480938745406,
                    2791.949219053383,
                    2828.9425287783106,
                    3003.3637643327074,
                    3127.113421778019,
                    3558.135685224518,
                    3835.461091718066,
                    4074.730916737977,
                    4080.5727188710407,
                    4359.9460338727495,
                    4407.110407081814,
                ],
                "load": {
                    "base": 13876.128382104485,
                    "heating": 0.0,
                    "cooling": 27406.821170457108,
                },
                "scores": [
                    {
                        "name": "r2",
                        "value": 0.9233878584644406,
                        "threshold": 0.75,
                        "ok": True,
                    },
                    {
                        "name": "cvrmse",
                        "value": 0.05562198622757366,
                        "threshold": 0.5,
                        "ok": True,
                    },
                ],
                "input_data": {
                    "X": [
                        31.5549430847168,
                        40.40647888183594,
                        40.545326232910156,
                        41.46236038208008,
                        45.7861213684082,
                        48.85377502441406,
                        59.53846740722656,
                        66.41313934326172,
                        72.34444427490234,
                        72.4892578125,
                        79.4146957397461,
                        80.5838623046875,
                    ],
                    "y": [
                        2771.7328796075267,
                        2866.628408060809,
                        2862.490733376347,
                        2778.9696134985625,
                        2841.454855450504,
                        2887.0038965747117,
                        3386.2120536070365,
                        3553.987575239154,
                        4136.844967333332,
                        3980.2125375555543,
                        4595.253582350758,
                        4622.158449907303,
                    ],
                    "sigma": None,
                    "absolute_sigma": None,
                },
                "nac": None,
            }
        ]
    }


@pytest.fixture
def adjusted_savings_fixture():
    adjusted_savings_result = {
        "adjusted_y": [1.0] * 12,
        "total_savings": 1.0,
        "average_savings": 1.0,
        "percent_savings": 1.0,
        "percent_savings_uncertainty": 1.0,
    }
    return {"confidence_interval": 0.8, "result": adjusted_savings_result}


@pytest.fixture
def normalized_saving_fixture():
    normalized_saving_result = {
        "normalized_y_pre": [1.0] * 12,
        "normalized_y_post": [1.0] * 12,
        "total_savings": 1.0,
        "average_savings": 1.0,
        "percent_savings": 1.0,
        "percent_savings_uncertainty": 1.0,
    }

    return {
        "X_pre": [1.0] * 12,
        "X_post": [1.0] * 12,
        "confidence_interval": 0.8,
        "result": normalized_saving_result,
    }


@pytest.fixture
def savings_fixture(
    changepointmodel_response_fixture,
    adjusted_savings_fixture,
    normalized_saving_fixture,
):
    cp_model = changepointmodel_response_fixture["results"][0]
    return {
        "results": [
            {
                "pre": cp_model,
                "post": cp_model,
                "adjusted_savings": adjusted_savings_fixture,
                "normalized_savings": normalized_saving_fixture,
            }
        ]
    }
