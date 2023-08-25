import pydantic
import pytest
import numpy as np

from changepointmodel.core.schemas import OneDimNDArrayField, NByOneNDArrayField
from pydantic import ConfigDict


def test_custom_fields_accept_lists_or_array_like():
    class MyModel(pydantic.BaseModel):
        arr: OneDimNDArrayField

        model_config = ConfigDict(arbitrary_types_allowed=True)

    m = MyModel(arr=[1, 2, 3, 4])

    assert isinstance(m.arr, np.ndarray)


def test_custom_fields_raise_if_value_arraylike_cannot_be_coerced_to_float():
    class MyModel(pydantic.BaseModel):
        arr: OneDimNDArrayField

        model_config = ConfigDict(arbitrary_types_allowed=True)

    with pytest.raises(pydantic.ValidationError):
        MyModel(
            arr=[
                "hep",
                "tup",
            ]
        )


# removed in 3.2
# def test_anybyanyndarrayfield_must_be_shape_of_m_by_n():
#     class MyModel(pydantic.BaseModel):
#         arr: AnyByAnyNDArrayField

#     with pytest.raises(pydantic.ValidationError):
#         MyModel(arr=[1.0, 2.0, 3.0])


def test_nbyonendarrayfield_must_contain_one_feature():
    class MyModel(pydantic.BaseModel):
        arr: NByOneNDArrayField

        model_config = ConfigDict(arbitrary_types_allowed=True)

    with pytest.raises(pydantic.ValidationError):
        MyModel(arr=[[1, 2, 3], [1, 2, 3]])

    MyModel(
        arr=[
            [
                1,
            ],
            [
                2,
            ],
        ]
    )


# removed 3.2
# def test_nbyonendarrayfield_must_by_shape_of_m_by_n():
#     class MyModel(pydantic.BaseModel):
#         arr: NByOneNDArrayField

#         class Config:
#             arbitrary_types_allowed = True

#     with pytest.raises(pydantic.ValidationError):
#         MyModel(arr=[1.0, 2.0, 3.0])

#     MyModel(
#         arr=[
#             [
#                 1,
#             ],
#             [
#                 1,
#             ],
#         ]
#     )


def test_nbyonendarrayfield_reshapes_data():
    class MyModel(pydantic.BaseModel):
        arr: OneDimNDArrayField

        model_config = ConfigDict(arbitrary_types_allowed=True)

    m = MyModel(arr=[1, 1, 1])

    assert list(m.arr) == [[1], [1], [1]]


def test_onedimndarrayfield_must_by_one_dimensional():
    class MyModel(pydantic.BaseModel):
        arr: OneDimNDArrayField

        model_config = ConfigDict(arbitrary_types_allowed=True)

    MyModel(arr=[1, 2, 3])

    with pytest.raises(pydantic.ValidationError):
        MyModel(
            arr=[
                [
                    1,
                ],
                [
                    2,
                ],
            ]
        )
