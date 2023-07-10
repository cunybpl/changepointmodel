from . import base, parameter_model as pm, coeffs_parser as cpar
from ..calc import models, bounds


def twop(
    name: str = "2P",
) -> pm.ParameterModelFunction[base.TwoParameterCallable, pm.TwoParameterModel]:
    """Since v3.1 Default factory method for a twop energy model.

    Args:
        name (str, optional): The name of the model. Defaults to "2P".

    Returns:
        ParameterModelFunction[TwoParameterCallable, TwoParameterModel]: A ParameterModelFunction instance that can be given to estimator.
    """
    return pm.ParameterModelFunction(
        name=name,
        f=models.twop,
        bounds=bounds.twop,
        parameter_model=pm.TwoParameterModel(),
        coefficients_parser=cpar.TwoParameterCoefficientParser(),
    )


def threepc(
    name: str = "3PC",
) -> pm.ParameterModelFunction[
    base.ThreeParameterCallable, pm.ThreeParameterCoolingModel
]:
    """Since v3.1 Default factory method for a threepc energy model.

    Args:
        name (str, optional): The name of the model. Defaults to "3PC".

    Returns:
        ParameterModelFunction[ThreeParameterCallable, ThreeParameterCoolingModel]: A ParameterModelFunction instance that can be given to estimator.
    """
    return pm.ParameterModelFunction(
        name=name,
        f=models.threepc,
        bounds=bounds.threepc,
        parameter_model=pm.ThreeParameterCoolingModel(),
        coefficients_parser=cpar.ThreeParameterCoefficientsParser(),
    )


def threeph(
    name: str = "3PH",
) -> pm.ParameterModelFunction[
    base.ThreeParameterCallable, pm.ThreeParameterHeatingModel
]:
    """Since v3.1 Default factory method for a threeph energy model.

    Args:
        name (str, optional): The name of the model. Defaults to "3PH".

    Returns:
        ParameterModelFunction[ThreeParameterCallable, ThreeParameterHeatingModel]: A ParameterModelFunction instance that can be given to estimator.
    """
    return pm.ParameterModelFunction(
        name=name,
        f=models.threeph,
        bounds=bounds.threeph,
        parameter_model=pm.ThreeParameterHeatingModel(),
        coefficients_parser=cpar.ThreeParameterCoefficientsParser(),
    )


def fourp(
    name: str = "4P",
) -> pm.ParameterModelFunction[base.FourParameterCallable, pm.FourParameterModel]:
    """Since v3.1 Default factory method for a fourp energy model.

    Args:
        name (str, optional): The name of the model. Defaults to "4P".

    Returns:
        ParameterModelFunction[FourParameterCallable, FourParameterModel]: A ParameterModelFunction instance that can be given to estimator.
    """
    return pm.ParameterModelFunction(
        name=name,
        f=models.fourp,
        bounds=bounds.fourp,
        parameter_model=pm.FourParameterModel(),
        coefficients_parser=cpar.FourParameterCoefficientsParser(),
    )


def fivep(
    name: str = "5P",
) -> pm.ParameterModelFunction[base.FiveParameterCallable, pm.FiveParameterModel]:
    """Since v3.1 Default factory method for a fivep energy model.

    Args:
        name (str, optional): The name of the model. Defaults to "5P".

    Returns:
        ParameterModelFunction[FiveParameterCallable, FiveParameterModel]: A ParameterModelFunction instance that can be given to the estimator
    """
    return pm.ParameterModelFunction(
        name=name,
        f=models.fivep,
        bounds=bounds.fivep,
        parameter_model=pm.FiveParameterModel(),
        coefficients_parser=cpar.FiveParameterCoefficientsParser(),
    )
