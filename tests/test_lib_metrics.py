import pytest  
import numpy as np
from changepointmodel.calc import metrics as energymodelmetrics

def test_r2_score_forwards_arguments(mocker): 

    y = np.array([1.,2.,3.])
    y_pred = np.array([4.,5.,6.])
    sample_weight = np.array([7.,8.,9.])

    mock = mocker.patch('sklearn.metrics.r2_score')
    energymodelmetrics.r2_score(y, y_pred, sample_weight=sample_weight)
    mock.assert_called_once_with(y, y_pred, sample_weight=sample_weight)


def test_rmse_score_forwards_arguments(mocker): 

    y = np.array([1.,2.,3.])
    y_pred = np.array([4.,5.,6.])
    sample_weight = np.array([7.,8.,9.])

    mock = mocker.patch('sklearn.metrics.mean_squared_error')
    energymodelmetrics.rmse(y, y_pred, sample_weight=sample_weight)
    mock.assert_called_once_with(y, y_pred, sample_weight=sample_weight, squared=False) # NOTE squared=False returns rmse according to skl docs



def test_cvrmse_score_forwards_arguments(mocker): 

    y = np.array([1.,2.,3.])
    y_pred = np.array([4.,5.,6.])
    sample_weight = np.array([7.,8.,9.])

    mock = mocker.patch('sklearn.metrics.mean_squared_error')
    energymodelmetrics.cvrmse(y, y_pred, sample_weight=sample_weight)
    mock.assert_called_once_with(y, y_pred, sample_weight=sample_weight, squared=False) 


def test_cvrmse_from_rmse(): 

    rmse = 2 
    y = np.array([1.,2.,3.])
    assert 1 == energymodelmetrics._cvrmse_from_rmse(rmse, y)

