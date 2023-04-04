from typing import List
from .models import EnergyChangepointModelResult
from changepointmodel.core.estimator import EnergyChangepointEstimator
from changepointmodel.core.pmodels import EnergyParameterModelT, ParamaterModelCallableT
from dataclasses import dataclass
from typing import Generic


@dataclass
class AppChangepointResultContainer(
    Generic[ParamaterModelCallableT, EnergyParameterModelT]
):
    """This is useful storage needed in the application to handle option c modeling methodology.
    The fit estimator must travel with its result through the pipeline.
    """

    estimator: EnergyChangepointEstimator[
        ParamaterModelCallableT, EnergyParameterModelT
    ]
    result: EnergyChangepointModelResult


AppChangepointResultContainers = List[
    AppChangepointResultContainer[ParamaterModelCallableT, EnergyParameterModelT]
]
