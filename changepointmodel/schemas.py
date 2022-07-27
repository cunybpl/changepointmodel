"""Defines types to use as inputs to CurvefitEstimator. This is agnostic of function since 
NumpyArray can be n-dimensional
"""
import numpy as np
from typing import Any, Dict, Optional, Union

import pydantic
from .nptypes import AnyByAnyNDArrayField, OneDimNDArrayField, AnyByAnyNDArray


class NpConfig: 
    json_encoders = {
        np.ndarray : lambda v: v.tolist()
    }


class CurvefitEstimatorDataModel(pydantic.BaseModel): 
    X: Union[OneDimNDArrayField, AnyByAnyNDArrayField]
    y: Optional[OneDimNDArrayField]   # NOTE this is optional so that different X values may be passed to a fit model
    sigma: Optional[OneDimNDArrayField]=None
    absolute_sigma: Optional[bool]=None


    @pydantic.validator('X')
    def validate_X(cls, v: AnyByAnyNDArray) -> AnyByAnyNDArray: # we don't want to overdo this check... but a basic reshape here is extremely helpful for processing
        if v.ndim == 1:  # assure 1d is reshaped according skl spec
            return v.reshape(-1, 1)        
        return np.atleast_2d(v)   # assure that anything else is at least 2d .. NOTE will not check for nested data... just know what your doing...


    @pydantic.root_validator
    def validate(cls, values: Dict[str, Any]) -> Dict[str, Any]: 

        if values['y'] is None and values['sigma'] is not None: 
            raise ValueError('Cannot pass `sigma` without `y`')

        if values['y'] is not None:  # if we are only passing X then we can skip validation
            
            xlen = len(values['X'])
            ylen = len(values['y'])

            if values['sigma'] is None:     
                if xlen != ylen: 
                    raise ValueError(f'X and y lengths do not match.')
            else: 
                siglen = len(values['sigma'])
                if not xlen == ylen == siglen: 
                    raise ValueError(f'X, y and sigma lengths to not match.')

        return values 

    class Config(NpConfig): ... 
        
# NOTE output data schemas are now just plain wrappers around internal dataclasses as per issue-12
# pydantic is only used as input validation for this lib... 

# class NormalizedSavingsResultData(pydantic.BaseModel): 

#     X_pre: NByOneNDArrayField
#     X_post: NByOneNDArrayField
#     confidence_interval: float
#     result: NormalizedSavingsResult

#     class Config(NpConfig): ... 



# class AdjustedSavingsResultData(pydantic.BaseModel): 
#     confidence_interval: float
#     result: AdjustedSavingsResult

#     class Config(NpConfig): ... 



# class EnergyChangepointModelResult(pydantic.BaseModel): 
#     name: str 
#     coeffs: EnergyParameterModelCoefficients
#     pred_y: OneDimNDArrayField
#     load: Optional[Load]=None
#     scores: Optional[List[Score]]=None
#     input_data: Optional[CurvefitEstimatorDataModel]=None
#     nac: Optional[PredictedSum]=None

#     class Config(NpConfig): ... 


        
# class SavingsResult(pydantic.BaseModel): 

#     pre: EnergyChangepointModelResult 
#     post: EnergyChangepointModelResult
#     adjusted_savings: AdjustedSavingsResultData
#     normalized_savings: Optional[NormalizedSavingsResultData]=None

#     class Config(NpConfig): ... 
        