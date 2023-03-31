from typing import List
from .models import EnergyChangepointModelResult
from changepointmodel.core.estimator import EnergyChangepointEstimator

from dataclasses import dataclass


# storage needed for option-c savings + filtering capabilities... estimator must move with result.
@dataclass
class BemaChangepointResultContainer(object):
    estimator: EnergyChangepointEstimator
    result: EnergyChangepointModelResult


BemaChangepointResultContainers = List[BemaChangepointResultContainer]
