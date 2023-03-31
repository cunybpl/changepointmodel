from changepointmodel.core import pmodels as models
from changepointmodel.core import loads
from changepointmodel.core.scoring import Scorer, R2, Cvrmse, ScoreEval
from changepointmodel.core.schemas import CurvefitEstimatorDataModel
from changepointmodel.core import factories
import numpy as np
from changepointmodel.core import calc

from .conftest import EnergyChangepointModelResultFactory


def test_energymodelfactory_integration_with_2p(generated_2p_data):
    Xdata = np.array(generated_2p_data["x"])
    ydata = np.array(generated_2p_data["y"])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)

    # configure correct model dependent handlers
    parser = models.TwoParameterCoefficientParser()
    twop_model = models.TwoParameterModel()
    cooling = loads.CoolingLoad()
    heating = loads.HeatingLoad()
    load_handler = loads.TwoParameterLoadHandler(twop_model, cooling, heating)

    # configure scoring
    cvrmse = ScoreEval(Cvrmse(), 0.5, lambda a, b: a < b)
    r2 = ScoreEval(R2(), 0.75, lambda a, b: a > b)
    evals = [r2, cvrmse]
    scorer = Scorer(
        evals
    )  # <<< NOTE I screwed this up we need to implement it like this....

    # note cannot use fit_one method here for estimator cause
    twop = factories.EnergyModelFactory.create(
        "2P", calc.models.twop, calc.bounds.twop, parser, twop_model, load_handler
    )
    estimator = twop.create_estimator()

    # X, y = estimator.sort_X_y(input_data.X, input_data.y)
    # # fit the changepoint model
    name, fitted_est = estimator.fit_one(estimator.model, input_data.X, input_data.y)
    load = twop.create_load_aggregator()
    result = EnergyChangepointModelResultFactory.create(fitted_est, load, scorer)

    # model
    assert result["name"] == "2P"

    # coeffs
    assert result["coeffs"].yint != 0.0
    assert result["coeffs"].changepoints == None
    assert len(result["coeffs"].slopes) == 1

    # loads
    assert result["load"].heating == 0.0
    assert result["load"].cooling != 0
    assert result["load"].base != 0

    # scores
    assert result["scores"][0].name == "r2"
    assert result["scores"][0].ok in (True, False)
    assert result["scores"][1].name == "cvrmse"
    assert result["scores"][1].ok in (True, False)

    # pred
    assert len(result["pred_y"])


def test_integration_with_3pc(generated_3pc_data):
    Xdata = np.array(generated_3pc_data["x"])
    ydata = np.array(generated_3pc_data["y"])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)

    # configure correct model dependent handlers
    parser = models.ThreeParameterCoefficientsParser()
    threep_model = models.ThreeParameterModel()
    cooling = loads.CoolingLoad()
    heating = loads.HeatingLoad()
    load_handler = loads.ThreeParameterLoadHandler(threep_model, cooling, heating)

    # configure scoring
    cvrmse = ScoreEval(Cvrmse(), 0.5, lambda a, b: a < b)
    r2 = ScoreEval(R2(), 0.75, lambda a, b: a > b)
    evals = [r2, cvrmse]
    scorer = Scorer(
        evals
    )  # <<< NOTE I screwed this up we need to implement it like this....

    # factory method to create an EnergyModel
    threepc = factories.EnergyModelFactory.create(
        "3PC",
        calc.models.threepc,
        calc.bounds.threepc,
        parser,
        threep_model,
        load_handler,
    )
    estimator = threepc.create_estimator()

    name, fitted_est = estimator.fit_one(estimator.model, input_data.X, input_data.y)
    load = threepc.create_load_aggregator()
    result = EnergyChangepointModelResultFactory.create(fitted_est, load, scorer)

    # model
    assert result["name"] == "3PC"

    # coeffs
    assert result["coeffs"].yint != 0.0
    assert len(result["coeffs"].changepoints) == 1
    assert len(result["coeffs"].slopes) == 1

    # loads
    assert result["load"].heating == 0.0
    assert result["load"].cooling != 0
    assert result["load"].base != 0

    # scores
    assert result["scores"][0].name == "r2"
    assert result["scores"][0].ok in (True, False)
    assert result["scores"][1].name == "cvrmse"
    assert result["scores"][1].ok in (True, False)

    # pred
    assert len(result["pred_y"])


def test_integration_with_3ph(generated_3ph_data):
    Xdata = np.array(generated_3ph_data["x"])
    ydata = np.array(generated_3ph_data["y"])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)

    # configure correct model dependent handlers
    parser = models.ThreeParameterCoefficientsParser()
    threep_model = models.ThreeParameterModel()
    cooling = loads.CoolingLoad()
    heating = loads.HeatingLoad()
    load_handler = loads.ThreeParameterLoadHandler(threep_model, cooling, heating)

    # configure scoring
    cvrmse = ScoreEval(Cvrmse(), 0.5, lambda a, b: a < b)
    r2 = ScoreEval(R2(), 0.75, lambda a, b: a > b)
    evals = [r2, cvrmse]
    scorer = Scorer(
        evals
    )  # <<< NOTE I screwed this up we need to implement it like this....

    # factory method to create an EnergyModel
    threeph = factories.EnergyModelFactory.create(
        "3PH",
        calc.models.threeph,
        calc.bounds.threeph,
        parser,
        threep_model,
        load_handler,
    )
    estimator = threeph.create_estimator()

    name, fitted_est = estimator.fit_one(estimator.model, input_data.X, input_data.y)
    load = threeph.create_load_aggregator()
    result = EnergyChangepointModelResultFactory.create(fitted_est, load, scorer)

    # model
    assert result["name"] == "3PH"

    # coeffs
    assert result["coeffs"].yint != 0.0
    assert len(result["coeffs"].changepoints) == 1
    assert len(result["coeffs"].slopes) == 1

    # loads
    assert result["load"].cooling == 0.0
    assert result["load"].heating != 0
    assert result["load"].base != 0

    # scores
    assert result["scores"][0].name == "r2"
    assert result["scores"][0].ok in (True, False)
    assert result["scores"][1].name == "cvrmse"
    assert result["scores"][1].ok in (True, False)

    # pred
    assert len(result["pred_y"])


def test_integration_with_4p(generated_4p_data):
    Xdata = np.array(generated_4p_data["x"])
    ydata = np.array(generated_4p_data["y"])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)

    # configure correct model dependent handlers
    parser = models.FourParameterCoefficientsParser()
    fourp_model = models.FourParameterModel()
    cooling = loads.CoolingLoad()
    heating = loads.HeatingLoad()
    load_handler = loads.FourParameterLoadHandler(fourp_model, cooling, heating)

    # configure scoring
    cvrmse = ScoreEval(Cvrmse(), 0.5, lambda a, b: a < b)
    r2 = ScoreEval(R2(), 0.75, lambda a, b: a > b)
    evals = [r2, cvrmse]
    scorer = Scorer(
        evals
    )  # <<< NOTE I screwed this up we need to implement it like this....

    # factory method to create an EnergyModel
    fourp = factories.EnergyModelFactory.create(
        "4P", calc.models.fourp, calc.bounds.fourp, parser, fourp_model, load_handler
    )
    estimator = fourp.create_estimator()

    name, fitted_est = estimator.fit_one(estimator.model, input_data.X, input_data.y)
    load = fourp.create_load_aggregator()
    result = EnergyChangepointModelResultFactory.create(fitted_est, load, scorer)

    # model
    assert result["name"] == "4P"

    # coeffs
    assert result["coeffs"].yint != 0.0
    assert len(result["coeffs"].changepoints) == 1
    assert len(result["coeffs"].slopes) == 2

    # loads
    assert result["load"].cooling != 0.0
    assert result["load"].heating != 0
    assert result["load"].base != 0

    # scores
    assert result["scores"][0].name == "r2"
    assert result["scores"][0].ok in (True, False)
    assert result["scores"][1].name == "cvrmse"
    assert result["scores"][1].ok in (True, False)

    # pred
    assert len(result["pred_y"])


def test_integration_with_5p(generated_4p_data):
    Xdata = np.array(generated_4p_data["x"])
    ydata = np.array(generated_4p_data["y"])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)

    # configure correct model dependent handlers
    parser = models.FiveParameterCoefficientsParser()
    fivep_model = models.FiveParameterModel()
    cooling = loads.CoolingLoad()
    heating = loads.HeatingLoad()
    load_handler = loads.FiveParameterLoadHandler(fivep_model, cooling, heating)

    # configure scoring
    cvrmse = ScoreEval(Cvrmse(), 0.5, lambda a, b: a < b)
    r2 = ScoreEval(R2(), 0.75, lambda a, b: a > b)
    evals = [r2, cvrmse]
    scorer = Scorer(
        evals
    )  # <<< NOTE I screwed this up we need to implement it like this....

    # factory method to create an EnergyModel
    fivep = factories.EnergyModelFactory.create(
        "5P", calc.models.fivep, calc.bounds.fivep, parser, fivep_model, load_handler
    )
    estimator = fivep.create_estimator()

    name, fitted_est = estimator.fit_one(estimator.model, input_data.X, input_data.y)
    load = fivep.create_load_aggregator()
    result = EnergyChangepointModelResultFactory.create(fitted_est, load, scorer)

    # model
    assert result["name"] == "5P"

    # coeffs
    assert result["coeffs"].yint != 0.0
    assert len(result["coeffs"].changepoints) == 2
    assert len(result["coeffs"].slopes) == 2

    # loads
    assert result["load"].cooling != 0.0
    assert result["load"].heating != 0
    assert result["load"].base != 0

    # scores
    assert result["scores"][0].name == "r2"
    assert result["scores"][0].ok in (True, False)
    assert result["scores"][1].name == "cvrmse"
    assert result["scores"][1].ok in (True, False)

    # pred
    assert len(result["pred_y"])
