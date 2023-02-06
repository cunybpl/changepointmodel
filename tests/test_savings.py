import pytest
import numpy as np
from changepointmodel.core.savings import AshraeAdjustedSavingsCalculator, \
    AshraeNormalizedSavingsCalculator, AdjustedSavingsResult, NormalizedSavingsResult, _cvrmse_score

from changepointmodel.core.calc.savings import adjusted, weather_normalized

def test_adjusted_savings_calls_adjusted_correctly(mocker): 

    class DummyEstimator(object):

        def len_y(self):
            return len(self.y)

        def total_y(self): 
            return sum(self.y)
                
    
    pre = DummyEstimator()

    pre.X = np.array([1.,])
    pre.y = np.array([2., ])
    pre.pred_y = np.array([3.,])
    pre.coeffs = (1,1,) 

    post = DummyEstimator()

    post.X = np.array([10.,])
    post.y = np.array([20.,])
    post.pred_y = np.array([30.,])
    post.coeffs = (1,1,1,)

    mocker.patch.object(_cvrmse_score, 'calc', return_value=0.42)
    mocker.patch('changepointmodel.core.savings._get_adjusted', return_value=np.array([42.,]))
    mock = mocker.patch('changepointmodel.core.calc.savings.adjusted', return_value=(1,1,1,1,))

    calc = AshraeAdjustedSavingsCalculator()
    calc.save(pre, post)

    mock.assert_called_once_with(42, 30, 0.42, 2, 1, 1, 0.8)


def test_adjusted_savings_calls_adjusted_correctly_with_kwargs(mocker): 

    class DummyEstimator(object):

        def len_y(self):
            return len(self.y)

        def total_y(self): 
            return sum(self.y)

        
    pre = DummyEstimator()

    pre.X = np.array([1.,])
    pre.y = np.array([2., ])
    pre.pred_y = np.array([3.,])
    pre.coeffs = (1,1,) 

    post = DummyEstimator()

    post.X = np.array([10.,])
    post.y = np.array([20.,])
    post.pred_y = np.array([30.,])
    post.coeffs = (1,1,1,)

    mocker.patch.object(_cvrmse_score, 'calc', return_value=0.42)
    mocker.patch('changepointmodel.core.savings._get_adjusted', return_value=np.array([42.,]))
    mock = mocker.patch('changepointmodel.core.calc.savings.adjusted', return_value=(1,1,1,1,))

    calc = AshraeAdjustedSavingsCalculator(confidence_interval=0.75, scalar=2)
    calc.save(pre, post)

    mock.assert_called_once_with(84, 60, 0.42, 2, 1, 1, 0.75)


def test_normalized_savings_calls_weather_normalzied_correctly(mocker): 

    class DummyEstimator(object):

        def len_y(self):
            return len(self.y)

        def total_y(self): 
            return sum(self.y)

        def predict(self, X): 
            return X

    pre = DummyEstimator()

    pre.X = np.array([1.,])
    pre.y = np.array([2., ])
    pre.pred_y = np.array([3.,])
    pre.coeffs = (1,1,) 

    post = DummyEstimator()

    post.X = np.array([10.,])
    post.y = np.array([20.,])
    post.pred_y = np.array([30.,])
    post.coeffs = (1,1,1,)

    mocker.patch.object(_cvrmse_score, 'calc', side_effect=[0.42, 0.43])
    mocker.patch('changepointmodel.core.savings._get_adjusted', return_value=np.array([42.,]))
    mock = mocker.patch('changepointmodel.core.calc.savings.weather_normalized', return_value=(1,1,1,1,))

    Xnorm = np.array([100.,]*12)

    calc = AshraeNormalizedSavingsCalculator(Xnorm)
    calc.save(pre, post)

    mock.assert_called_once_with(1200.0, 1200.0, 0.42, 0.43, 1, 1, 2, 3, 12, .8)


def test_normalized_savings_calls_weather_normalzied_correctly_with_kwargs(mocker): 

    class DummyEstimator(object):
        
        def len_y(self):
            return len(self.y)

        def total_y(self): 
            return sum(self.y)
        

        def predict(self, X): 
            return X

    pre = DummyEstimator()

    pre.X = np.array([1.,])
    pre.y = np.array([2., ])
    pre.pred_y = np.array([3.,])
    pre.coeffs = (1,1,) 

    post = DummyEstimator()

    post.X = np.array([10.,])
    post.y = np.array([20.,])
    post.pred_y = np.array([30.,])
    post.coeffs = (1,1,1,)

    mocker.patch.object(_cvrmse_score, 'calc', side_effect=[0.42, 0.43])
    mocker.patch('changepointmodel.core.savings._get_adjusted', return_value=np.array([42.,]))
    mock = mocker.patch('changepointmodel.core.calc.savings.weather_normalized', return_value=(1,1,1,1,))

    Xnorm = np.array([100.,]*12)

    calc = AshraeNormalizedSavingsCalculator(Xnorm, confidence_interval=0.95, scalar=2)
    calc.save(pre, post)

    mock.assert_called_once_with(2400.0, 2400.0, 0.42, 0.43, 1, 1, 2, 3, 12, .95)

