from typing import List
from .models import EnergyChangepointModelResult
from changepointmodel.core.estimator import EnergyChangepointEstimator
from changepointmodel.core.pmodels import EnergyParameterModelT, ParamaterModelCallableT
from dataclasses import dataclass
from typing import Generic


# storage needed for option-c savings + filtering capabilities... estimator must move with result.
@dataclass
class BemaChangepointResultContainer(
    Generic[ParamaterModelCallableT, EnergyParameterModelT]
):
    estimator: EnergyChangepointEstimator[
        ParamaterModelCallableT, EnergyParameterModelT
    ]
    result: EnergyChangepointModelResult


BemaChangepointResultContainers = List[
    BemaChangepointResultContainer[ParamaterModelCallableT, EnergyParameterModelT]
]
