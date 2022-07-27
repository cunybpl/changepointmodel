""" APIs for handling adjusted and normalized savings as per ashrae.
"""
from dataclasses import dataclass
import numpy as np
from .nptypes import NByOneNDArray, OneDimNDArray, OneDimNDArrayField
from .estimator import EnergyChangepointEstimator
from .calc import savings as libsavings

from .scoring import Cvrmse
_cvrmse_score = Cvrmse()

import abc 

# I have to pass the pydantic ndarray objects as fields so 
# that we can validate the data later.

def _get_adjusted(
    pre: EnergyChangepointEstimator, 
    post: EnergyChangepointEstimator) -> OneDimNDArray:
    
    return pre.adjust(post)


@dataclass 
class AdjustedSavingsResult(object): 
    adjusted_y: OneDimNDArrayField
    total_savings: float 
    average_savings: float
    percent_savings: float 
    percent_savings_uncertainty: float 


@dataclass 
class NormalizedSavingsResult(object): 
    normalized_y_pre: OneDimNDArrayField
    normalized_y_post: OneDimNDArrayField
    total_savings: float
    average_savings : float 
    percent_savings : float 
    percent_savings_uncertainty : float 


class ISavingsCalculator(abc.ABC): 
    
    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator) -> AdjustedSavingsResult: ...
    

class AbstractSavings(abc.ABC):

    def __init__(self, confidence_interval: float=0.80):
        """A Savings model calculates either adjusted or weather normalized savings 
        based on ashrae formulas and methodology. Should be used in the context of option-c 
        reporting with changepoint models.

        Args:
            confidence_interval (float, optional): The confidence interval to be used for the calculations. Defaults to 0.80.
        """
        self._confidence_interval = confidence_interval

    @property 
    def confidence_interval(self): 
        return self._confidence_interval

class AshraeAdjustedSavingsCalculator(AbstractSavings, ISavingsCalculator): 
    
    
    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator) -> AdjustedSavingsResult:
        """Controller method for calculated AdjustedSavings values and uncertainiies.

        Args:
            pre (EnergyChangepointEstimator): A fitted model for the pre retrofit period
            post (EnergyChangepointEstimator): A fitted model for the post retrofit period

        Returns:
            AdjustedSavingsResult: The AdjustedSavings result.
        """

        adjusted_pred_y = _get_adjusted(pre, post) # XXX this is expensive... :/ 
        pre_cvrmse = _cvrmse_score(pre.y, pre.pred_y) # XXX this is expensive... :/ 
        pre_p = len(pre.coeffs)
        
        pre_n = pre.len_y()
        post_n = post.len_y() 
                 
        gross_adjusted_y = np.sum(adjusted_pred_y)  
        gross_post_y = post.total_y()      

        savings = libsavings.adjusted(
            gross_adjusted_y, 
            gross_post_y, 
            pre_cvrmse, 
            pre_p, 
            pre_n, 
            post_n, 
            self._confidence_interval)

        total_savings, average_savings, percent_savings, percent_savings_uncertainty = savings
        return AdjustedSavingsResult(
            adjusted_pred_y,
            total_savings, 
            average_savings, 
            percent_savings, 
            percent_savings_uncertainty
        )

# This calculation must handle different X values. Design wise its easier to pass the X vals into the constructor. 
# This way we can pass configured objects to any factory method as opposed to passing data directly

class AshraeNormalizedSavingsCalculator(AbstractSavings, ISavingsCalculator): 
    
    def __init__(self, X_norms: NByOneNDArray, confidence_interval: float=0.80):
        """The Normalized savings calculations provide pre and post X arrays. These are used within the context
        of weather normalized savings for option-c retrofits.

        Args:
            X_norms (NByOneNDArray): Normalized X data for pre-retrofit and post-retrofit related normalized calculation.
            confidence_interval (float, optional): The confidence interval for the uncertainity calculations. Defaults to 0.80.
        """
        super().__init__(confidence_interval=confidence_interval)
        self._X_norms = X_norms

    @property 
    def X_norms(self):
        return self._X_norms 

    def save(self, 
        pre: EnergyChangepointEstimator, 
        post: EnergyChangepointEstimator,  
        ) -> NormalizedSavingsResult:  
        """The controller method for the normalized savings calculation.

        Args:
            pre (EnergyChangepointEstimator): The fitted pre retrofit model.
            post (EnergyChangepointEstimator): The fitted post retrofit model.

        Returns:
            NormalizedSavingsResult: The result of the normalized savings calculation
        """

        # setup
        normalized_pred_y_pre = pre.predict(self._X_norms)   # XXX expensive :(
        normalized_pred_y_post = post.predict(self._X_norms)
        
        gross_normalized_pred_y_pre = np.sum(normalized_pred_y_pre)
        gross_normalized_pred_y_post = np.sum(normalized_pred_y_post)

        pre_cvrmse = _cvrmse_score(pre.y, pre.pred_y)  # XXX expensive :( -- NOTE also this is from the original model on the actual X
        post_cvrmse = _cvrmse_score(post.y, post.pred_y)

        pre_n = pre.len_y()
        post_n = post.len_y() 

        pre_p = len(pre.coeffs)
        post_p = len(post.coeffs)

        n_norm = len(self._X_norms)

        savings = libsavings.weather_normalized(
            gross_normalized_pred_y_pre, 
            gross_normalized_pred_y_post, 
            pre_cvrmse, 
            post_cvrmse, 
            pre_n, 
            post_n, 
            pre_p, 
            post_p, 
            n_norm,
            self._confidence_interval)
        
        total_savings, average_savings, percent_savings, percent_savings_uncertainty = savings

        return NormalizedSavingsResult(
            normalized_pred_y_pre, 
            normalized_pred_y_post, 
            total_savings, 
            average_savings, 
            percent_savings, 
            percent_savings_uncertainty)
        