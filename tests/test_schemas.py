from typing import List
from changepointmodel import schemas 

import numpy as np
import pytest
import pydantic
import dataclasses

from changepointmodel.nptypes import NByOneNDArrayField, OneDimNDArrayField, AnyByAnyNDArrayField


def test_curvefitestimatordatamodel_handles_1d_xdata(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    schemas.CurvefitEstimatorDataModel(X=xdata)

    
def test_curvefittestimatordatamodel_handles_2d_xdata(): 

    xdata = np.array([[1.], [2.], [3.], [4.], [5.]])
    schemas.CurvefitEstimatorDataModel(X=xdata)


def test_curvefittestimatordatamodel_transforms_1d_xdata_to_2d_xdata(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    d = schemas.CurvefitEstimatorDataModel(X=xdata) 
    
    test = np.array([[1.], [2.], [3.], [4.], [5.]])
    assert test.shape == d.X.shape


def test_curvefitestimatordatamodel_handles_optional_data(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    ydata = np.array([1., 2., 3., 4., 5.])
    sigma = np.array([1., 2., 3., 4., 5.])

    schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma, absolute_sigma=False)


def test_curvefitestimatordatamodel_raises_validationerror_on_len_mismatch(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    ydata = np.array([1., 2., 3., 4.])
    sigma = np.array([1., 2., 3., 4., 5.])

    # check various combos of len mismatch
    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)
    
    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)
    
    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)

    with pytest.raises(pydantic.ValidationError): 
        ydata = np.array([1., 2., 3., 4., 5.])
        sigma = np.array([1., 2., 3., 4.])
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)
    
    with pytest.raises(pydantic.ValidationError): 
        ydata = np.array([1., 2., 3., 4.])
        sigma = np.array([1., 2., 3.])
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)


def test_curvefitestimatordatamodel_raises_validationerror_if_sigma_not_y(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    sigma = np.array([1., 2., 3., 4., 5.])

    with pytest.raises(pydantic.ValidationError): 
        schemas.CurvefitEstimatorDataModel(X=xdata, sigma=sigma)


def test_curvefitestimatordatamodel_returns_valid_json(): 

    xdata = np.array([1., 2., 3., 4., 5.])
    ydata = np.array([1., 2., 3., 4., 5.])
    
    d = schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)
    d.json()




def test_openapi_schemas_are_correctly_generated_for_custom_nptypes(): 

    @dataclasses.dataclass 
    class Check:
        a: OneDimNDArrayField 
        b: NByOneNDArrayField
        c: AnyByAnyNDArrayField  
    

    class CheckModel(pydantic.BaseModel): 
        check : Check
        thing : int 
        mylist : List[float]
        
        class Config:
            json_encoders = {
                np.ndarray : lambda v: v.tolist()
            }

    check = Check(
        a=[1,2,3], 
        b=[[1,],[2,],[3,],], 
        c=[[1,2,3],[4,5,6]], 
        )
    CheckModel(check=check, thing=42, mylist=[7,8,9])
    schema = CheckModel.schema()


    assert 'a' in schema['definitions']['Check']['properties']
    assert "b" in schema['definitions']['Check']['properties']
    assert "c" in schema['definitions']['Check']['properties']

    a = {"title": "A","type": "array","items": {"type": "number"}}
    b = {"title": "B","type": "array","items": {"type": "array","items": {"type": "number"}}}
    c = {"title": "C","type": "array","items": {"type": "array","items": {"type": "number"}}}

    assert schema['definitions']['Check']['properties']['a'] == a 
    assert schema['definitions']['Check']['properties']['b'] == b 
    assert schema['definitions']['Check']['properties']['c'] == c 
