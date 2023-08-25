from typing import List
from changepointmodel.core import schemas

import numpy as np
import pytest
import pydantic
import dataclasses


from pydantic import ConfigDict


def test_curvefitestimatordatamodel_handles_1d_xdata():
    xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)


def test_curvefittestimatordatamodel_handles_2d_xdata():
    xdata = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)


def test_curvefittestimatordatamodel_transforms_1d_xdata_to_2d_xdata():
    xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    d = schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)

    test = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    assert test.shape == d.X.shape


def test_curvefitestimatordatamodel_handles_optional_data():
    xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    schemas.CurvefitEstimatorDataModel(
        X=xdata, y=ydata, sigma=sigma, absolute_sigma=False
    )


def test_curvefitestimatordatamodel_raises_validationerror_on_len_mismatch():
    xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ydata = np.array([1.0, 2.0, 3.0, 4.0])
    sigma = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # check various combos of len mismatch
    with pytest.raises(pydantic.ValidationError):
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)

    with pytest.raises(pydantic.ValidationError):
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)

    with pytest.raises(pydantic.ValidationError):
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)

    with pytest.raises(pydantic.ValidationError):
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma = np.array([1.0, 2.0, 3.0, 4.0])
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)

    with pytest.raises(pydantic.ValidationError):
        ydata = np.array([1.0, 2.0, 3.0, 4.0])
        sigma = np.array([1.0, 2.0, 3.0])
        schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata, sigma=sigma)


def test_curvefitestimatordatamodel_returns_valid_json():
    xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    d = schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)
    d.json()


def test_openapi_schemas_are_correctly_generated_for_custom_nptypes():
    @dataclasses.dataclass
    class Check:
        a: schemas.OneDimNDArrayField
        b: schemas.NByOneNDArrayField

        model_config = ConfigDict(arbitrary_types_allowed=True)

    class CheckModel(pydantic.BaseModel):
        check: Check
        thing: int
        mylist: List[float]
        # TODO[pydantic]: The following keys were removed: `json_encoders`.
        # Check https://docs.pydantic.dev/dev-v2/migration/#changes-to-config for more information.
        model_config = ConfigDict(
            json_encoders={np.ndarray: lambda v: v.tolist()},
            arbitrary_types_allowed=True,
        )

    check = Check(
        a=[1, 2, 3],
        b=[
            [
                1,
            ],
            [
                2,
            ],
            [
                3,
            ],
        ],
    )
    CheckModel(check=check, thing=42, mylist=[7, 8, 9])
    schema = CheckModel.model_json_schema()

    print(schema)

    assert "a" in schema["$defs"]["Check"]["properties"]
    assert "b" in schema["$defs"]["Check"]["properties"]

    a = {"title": "A", "type": "array", "items": {"type": "number"}}
    b = {
        "title": "B",
        "type": "array",
        "items": {"type": "array", "items": {"type": "number"}},
    }

    assert schema["$defs"]["Check"]["properties"]["a"] == a
    assert schema["$defs"]["Check"]["properties"]["b"] == b


def test_schema_returns_sorted_X_y():
    xdata = np.array([3.0, 5.0, 1.0, 2.0, 4.0])
    ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    d = schemas.CurvefitEstimatorDataModel(X=xdata, y=ydata)

    X, y, idx = d.sorted_X_y()

    assert [list(x) for x in X] == [[1.0], [2.0], [3.0], [4.0], [5.0]]
    assert list(y) == [3.0, 4.0, 1.0, 5.0, 2.0]
    assert list(idx) == [2, 3, 0, 4, 1]
