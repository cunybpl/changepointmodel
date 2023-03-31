from changepointmodel.core import pmodels, loads
from changepointmodel.core.calc import models as cpmodels
from changepointmodel.core.calc import bounds as cpbounds
from changepointmodel.core.estimator import EnergyChangepointEstimator
from changepointmodel.core.loads import EnergyChangepointLoadsAggregator
from changepointmodel.core.factories import EnergyModelFactory
from typing import List, Tuple


# NOTE that this part of the API should be thread-safe since the underlying objects in EnergyModels that these factories
# produce are stateless
_cooling = loads.CoolingLoad()
_heating = loads.HeatingLoad()
_base = loads.Baseload()

_twop_parser = pmodels.TwoParameterCoefficientParser()
_twop_model = pmodels.TwoParameterModel()
_twop_load_handler = loads.TwoParameterLoadHandler(
    _twop_model, _cooling, _heating, _base
)

_threep_parser = pmodels.ThreeParameterCoefficientsParser()
_threep_model = pmodels.ThreeParameterModel()
_threep_load_handler = loads.ThreeParameterLoadHandler(
    _threep_model, _cooling, _heating, _base
)

_fourp_parser = pmodels.FourParameterCoefficientsParser()
_fourp_model = pmodels.FourParameterModel()
_fourp_load_handler = loads.FourParameterLoadHandler(
    _fourp_model, _cooling, _heating, _base
)

_fivep_parser = pmodels.FiveParameterCoefficientsParser()
_fivep_model = pmodels.FiveParameterModel()
_fivep_load_handler = loads.FiveParameterLoadHandler(
    _fivep_model, _cooling, _heating, _base
)


_twop_changepoint_model = EnergyModelFactory.create(
    "2P", cpmodels.twop, cpbounds.twop, _twop_parser, _twop_model, _twop_load_handler
)
_threepc_changepoint_model = EnergyModelFactory.create(
    "3PC",
    cpmodels.threepc,
    cpbounds.threepc,
    _threep_parser,
    _threep_model,
    _threep_load_handler,
)
_threeph_changepoint_model = EnergyModelFactory.create(
    "3PH",
    cpmodels.threeph,
    cpbounds.threeph,
    _threep_parser,
    _threep_model,
    _threep_load_handler,
)
_fourp_changepoint_model = EnergyModelFactory.create(
    "4P",
    cpmodels.fourp,
    cpbounds.fourp,
    _fourp_parser,
    _fourp_model,
    _fourp_load_handler,
)
_fivep_changepoint_model = EnergyModelFactory.create(
    "5P",
    cpmodels.fivep,
    cpbounds.fivep,
    _fivep_parser,
    _fivep_model,
    _fivep_load_handler,
)


_models = {
    "2P": _twop_changepoint_model,
    "3PC": _threepc_changepoint_model,
    "3PH": _threeph_changepoint_model,
    "4P": _fourp_changepoint_model,
    "5P": _fivep_changepoint_model,
}


def get_changepoint_model_pair(
    name: str,
) -> Tuple[EnergyChangepointEstimator, EnergyChangepointLoadsAggregator]:
    """Returns a Tuple of configured instances of EnergyChangepointEstimator
    and LoadsAggregator for modeling. This must be called to make sure the rest of the running modeling code is
    safe.

    Args:
        name (str): name of the model.

    Returns:
        Tuple[EnergyChangepointEstimator, EnergyChangepointLoadsAggregator]: _description_
    """
    m = _models.get(name)
    return m.create_estimator(), m.create_load_aggregator()
