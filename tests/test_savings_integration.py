from ashrae import parameter_models as models
from ashrae import loads
from ashrae.scoring import Scorer, R2, Cvrmse, ScoreEval 
from ashrae.schemas import CurvefitEstimatorDataModel
from ashrae import factories
import numpy as np
from ashrae import lib
from ashrae import savings

def test_savings_integration_with_pre_and_post(generated_3pc_data, generated_4p_post_data):
    Xdata = np.array(generated_3pc_data['x']) 
    ydata = np.array(generated_3pc_data['y'])
    Xdata_post = np.array(generated_4p_post_data['x']) 
    ydata_post = np.array(generated_4p_post_data['y'])
    input_data = CurvefitEstimatorDataModel(X=Xdata, y=ydata)
    input_data_post =  CurvefitEstimatorDataModel(X=Xdata_post, y=ydata_post)

    # configure correct model dependent handlers for 3pc
    parser_3pc = models.ThreeParameterCoefficientsParser()
    threep_model = models.ThreeParameterModel()
    cooling = loads.CoolingLoad() 
    heating = loads.HeatingLoad()
    load_handler_3pc = loads.ThreeParameterLoadHandler(threep_model, cooling, heating)

    # configure correct model dependent handlers for 4p
    parser_4p = models.FourParameterCoefficientsParser()
    fourp_model = models.FourParameterModel()
    load_handler_4p = loads.FourParameterLoadHandler(fourp_model, cooling, heating)

    # configure scoring 
    cvrmse = ScoreEval(Cvrmse(), 0.5, lambda a,b: a < b) 
    r2 = ScoreEval(R2(), 0.75, lambda a,b: a > b) 
    evals = [r2, cvrmse]
    scorer = Scorer(evals)   # <<< NOTE I screwed this up we need to implement it like this.... 


    #note cannot use fit_one method here for estimator cause
    threepc = factories.EnergyModelFactory.create('3PC', lib.models.threepc, lib.bounds.threepc, parser_3pc, threep_model, load_handler_3pc)
    estimator_pre = threepc.create_estimator()

    fourp = factories.EnergyModelFactory.create('4P', lib.models.fourp, lib.bounds.fourp, parser_4p, fourp_model, load_handler_4p)
    estimator_post = fourp.create_estimator()

    # X, y = estimator.sort_X_y(input_data.X, input_data.y)
    # # fit the changepoint model 
    name_pre, fitted_est_pre = estimator_pre.fit_one(estimator_pre.model, input_data.X, input_data.y)
    name_post, fitted_est_post =  estimator_post.fit_one(estimator_post.model, input_data_post.X, input_data_post.y)

    adjusted_saving = savings.AshraeAdjustedSavingsCalculator()
    normalized_saving = savings.AshraeNormalizedSavingsCalculator(input_data.X, input_data_post.X)

    savings_result = factories.SavingsResultFactory.create(fitted_est_pre, fitted_est_post,
                                            adjusted_saving, normalized_saving, threepc.load, fourp.load, scorer
                                            )
    
    #pre basic check
    assert savings_result.pre.name == '3PC'
    assert savings_result.pre.scores[0].ok == True
    assert savings_result.pre.scores[1].ok == True

    #post basic check
    assert savings_result.post.name == '4P'
    assert savings_result.post.scores[0].ok == True
    assert savings_result.post.scores[1].ok == True

    #adujusted savings
    assert savings_result.adjusted_savings.confidence_interval == 0.8
    assert savings_result.adjusted_savings.result.total_savings

    #normalized savings
    assert savings_result.normalized_savings.confidence_interval == 0.8
    assert savings_result.normalized_savings.result.total_savings