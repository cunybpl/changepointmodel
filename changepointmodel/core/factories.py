"""Some factory methods for configuration and generating result pydantic schemas to and from internals.

Some of the configuration is complicated to produce the vast amount of results we require along with a model. (We used to call this the post-modeling calcs at BPL)
These are here as higher level abstractions that help keep all the library components loosely coupled but still able to work together. While we are take some 
steps to make sure the configuration between the various components are valid it is largely at this time still up to the caller to make sure they are using and 
testing a correct configuration.

These Factory objects do not make any assumptions about workflow and should be wrapped 
in even higher order functions or classes on in application (i.e. an RPC service or batch processing script). 
"""


from .estimator import EnergyChangepointEstimator
from typing import Union, Generic
from . import loads
from dataclasses import dataclass
from . import pmodels as pmodels

from .pmodels import EnergyParameterModelT, ParamaterModelCallableT


class EnergyModelConfigurationError(TypeError):
    """raised if an EnergyModel is constructed with the wrong load_handler"""


@dataclass
class EnergyModel(Generic[ParamaterModelCallableT, EnergyParameterModelT]):
    """The purpose of this container object is to keep the correct ParameterModelFunction
    and LoadHandler in the same place.
    """

    model: pmodels.ParameterModelFunction[
        ParamaterModelCallableT, EnergyParameterModelT
    ]
    load_handler: loads.AbstractLoadHandler[EnergyParameterModelT]

    def create_estimator(
        self,
    ) -> EnergyChangepointEstimator[ParamaterModelCallableT, EnergyParameterModelT]:
        """Spawn a new estimator from the model.

        Returns:
            EnergyChangepointEstimator: An instance
        """
        return EnergyChangepointEstimator(model=self.model)

    def create_load_aggregator(
        self,
    ) -> loads.EnergyChangepointLoadsAggregator[EnergyParameterModelT]:
        """Convenience method to get a reference to this model's load.

        XXX I added this to create a public API since this part of the object might change.

        Returns:
            loads.EnergyChangepointLoadsAggregator: An Aggregator that initializes the handler.
        """
        return loads.EnergyChangepointLoadsAggregator(handler=self.load_handler)


class EnergyModelFactory(Generic[ParamaterModelCallableT, EnergyParameterModelT]):
    @classmethod
    def create(
        cls,
        name: str,
        f: ParamaterModelCallableT,
        b: Union[pmodels.BoundCallable, pmodels.Bound],
        parser: pmodels.ICoefficientParser,
        parameter_model: EnergyParameterModelT,
        load_handler: loads.AbstractLoadHandler[EnergyParameterModelT],
    ) -> EnergyModel[ParamaterModelCallableT, EnergyParameterModelT]:
        """Construct an model and a loads factory simultaneously.
        Creates a convenient container object which will help keep model dependent
        calculations together within more complicated workflows.

        Args:
            name (str): [description]
            f (pmodels.ModelCallable): [description]
            b (Union[pmodels.BoundCallable, pmodels.Bound]): [description]
            parser (pmodels.ICoefficientParser): [description]
            parameter_model (pmodels.EnergyParameterModel): [description]
            load_handler (loads.AbstractLoadHandler): [description]

        Raises:
            EnergyModelConfigurationError(TypeError): If the parameter model and load_handler values are incorrect.

        Returns:
            EnergyModel: [description]
        """

        model = pmodels.ParameterModelFunction(name, f, b, parameter_model, parser)

        if not isinstance(
            model.parameter_model, load_handler.model.__class__
        ):  # we can at least check this one so we do... this is similar to loads.py
            raise EnergyModelConfigurationError(
                f"parameter_model and load_handler models do not match: {parameter_model.__class__}, {load_handler.model.__class__}"
            )

        return EnergyModel(model=model, load_handler=load_handler)
