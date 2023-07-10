from ._version import VERSION

__version__ = VERSION

# NOTE from 3.1 I am exposing these as top level packages

from changepointmodel.core import calc, estimator, savings, schemas, utils

# Importing from here will give you the 3.1 factory methods at the top level
# for convenience.
from changepointmodel.core.pmodels import factories

__all__ = (
    "calc",
    "estimator",
    "savings",
    "schemas",
    "utils",
    "factories",
)
