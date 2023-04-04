""" 
A set of pydantic models for validating input, output and configuration options. 

The "Result" models come equipped with various methods for parsing and flattening the inner result 
data into python primitives. These can be used to insert into databases or write to disk. 

TODO many of the parse methods should be marked as protected and better documented
"""

from changepointmodel.core.nptypes import (
    OneDimNDArrayField,
    NByOneNDArrayField,
    AnyByAnyNDArrayField,
)
import pydantic
from typing import List, Optional, Union, Any, Dict, Tuple, Annotated
import numpy as np
from changepointmodel.core.schemas import NpConfig
from pydantic import Field
import enum


SklScoreReturnType = Union[float, AnyByAnyNDArrayField, Any]
GenericRecords = List[Dict[str, Any]]


class FilterHowEnum(enum.Enum):
    """How to filter your models. Details can be found in `changepointmodel.bema.filter_`."""

    best_score = "best_score"
    threshold_ok = "threshold_ok"
    threshold_ok_first_is_best = "threshold_ok_first_is_best"


class FilterWhichEnum(enum.Enum):
    """Which score to use to filter your models. Details can be found in `changepointmodel.bema.filter_`."""

    r2 = "r2"
    cvrmse = "cvrmse"


class UsageTemperatureData(pydantic.BaseModel):
    """Provide the oat and usage data for energy changepointmodel modeling. We validate on array size and
    shape. This is timescale agnostic but both oat and usage must be the same size.

    """

    oat: List[float] = Field(
        ...,
        description="Outside air temperature for changepoint regression. Should be same len as usage.",
    )
    usage: List[float] = Field(
        ...,
        description="Usage points for changepoint regression. Should be same len as oat.",
    )

    @pydantic.validator("oat")
    def validate_oat(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("oat cannot be empty")
        return v

    @pydantic.validator("usage")
    def validate_usage(cls, v: List[Any], values: Dict[str, Any]) -> List[float]:
        if len(v) == 0:
            raise ValueError("usage cannot be empty")
        if "oat" in values and len(v) != len(values["oat"]):
            raise ValueError("sizes of oat and usage arrays are mismatched.")
        return v


class FilterConfig(pydantic.BaseModel):
    """Filter configuration options. While this is entirely optional it is useful especially for
    batch processing. Details for how filtering works can be found in `changepointmodel.bema.filter_` module.
    """

    which: FilterWhichEnum = Field(
        default=FilterWhichEnum.r2, description="Filter on r2 or cvrmse value."
    )
    how: FilterHowEnum = Field(
        default=FilterHowEnum.threshold_ok_first_is_best,
        description="How to filter the changepointmodels. Default is threshold_ok_first_is_best.",
    )
    extras: bool = Field(
        default=False,
        description="Flag to filter models through data population test, shape test and t-test.",
    )


Threshold = Annotated[float, pydantic.confloat(ge=0, le=1)]


# filtered to calcuale savings, but return all the models
class ChangepointModelConfig(pydantic.BaseModel):
    """Which models to attempt to run and what scores to report on. We do not filter without an explicit
    FilterConfig object. Note that the order of models will affect the filtering results if the option is
    given as `threshold_ok_first_is_best`.
    """

    models: List[str] = Field(
        ..., description="Model types to attempt. Options are 5P, 4P, 3PC, 3PH, 2P."
    )
    r2_threshold: Threshold = Field(
        0.75,
        description="The r2 threshold to report. After modeling we will analyze the r2 and see if it is gte this number.",
    )
    cvrmse_threshold: Threshold = Field(
        0.25,
        description="The cvrmse threshold. After modeling we will analyze the cvrmse value and see if it lte this number.",
    )
    model_filter: Optional[FilterConfig]

    @pydantic.validator("models")
    def validate_models_name(cls, v: List[str]) -> List[str]:
        for i in v:
            if i not in ["2P", "3PC", "3PH", "4P", "5P"]:
                raise ValueError(f"{i} not a valid model")
        return v


class EnergyChangepointModelRequest(pydantic.BaseModel):
    """This is the top level request object. A non-zero threshold is provided to validate
    whether there are enough points to model. Note that we also do not accept
    NULL or NaN values for X or y data since it will cause modeling errors in scipy.
    """

    nonzero_threshold: Threshold = Field(
        0.8,
        description="Threshold for percent of number of non-zero input data points.",
    )
    model_config: ChangepointModelConfig = Field(
        ..., description="A configuration for this modeling request."
    )
    usage: UsageTemperatureData = Field(
        ..., description="Usage and temperature data for this modeling request."
    )

    @pydantic.validator("usage")
    def validate_usage_threshold(
        cls, v: Any, values: Dict[str, Any]
    ) -> Any:  # usage UsageTemperatureData will get validated first
        def _validate_with_threshold(value_ls: List[float]) -> None:
            zeros = [i for i in value_ls if i == 0.0]
            percent_non_zeros = 1 - (len(zeros) / len(value_ls))
            if percent_non_zeros < values["nonzero_threshold"]:
                raise ValueError(
                    "number of non-zeros data point exceeds allowed threshold."
                )

        # _validate_with_threshold(v.oat) -- see #211
        _validate_with_threshold(v.usage)
        return v


class BaselineChangepointModelRequest(EnergyChangepointModelRequest):
    """A subclass of EnergyChangepointmodel request that accepts normalized data. Note that
    we will accept monthly(12), hourly(24) or daily(365) points for this field. It should be the same
    timescale as X and y data that you wish to model. We do not validate this aspect of modeling. It is up
    to the user to make sure their data is correct before trying to model.
    """

    norms: Optional[List[float]] = Field(
        None,
        description="Optional normalized outside air temperature data. Used to calculate NAC. Length can be 12 (monthly), 365 (daily) or 24 (hourly).",
    )

    @pydantic.validator("norms")
    def validate_norms(cls, v: List[float]) -> List[float]:
        if v and len(v) not in [12, 24, 365]:
            raise ValueError(
                "Invalid length of normalized temperature data. Should be 12 (monthly), 24 (hourly) or 365 (daily)."
            )
        return v


ConfidenceInterval = Annotated[float, pydantic.confloat(le=1, ge=0.5)]
Scalar = Annotated[float, pydantic.confloat(ge=1.0)]


class SavingsRequest(pydantic.BaseModel):
    """A request for option-c savings. It is essentially composed of 2 EnergyChangepointModelRequsts, one for pre and post.

    For this application we do our best to validate that you are strictly conforming to monthly aggregates by checking for 12 pre and post
    points. This is to assure that the resulting savings calculations will produce the correct values since they are tuned specifically
    for monthly calculations. If a more timescale agnostic approach to performing these calculations is determined we will
    design a more robust savings request.

    The Ashrae calculations themselves can be found in `changepointmodel.core.calc.savings` and `changepointmodel.core.calc.uncertainties`.
    We were careful to site references when we developed the library.

    The confidence interval for savings calculations is given here as well as scalar. The scalar is mainly used to assure that any
    per/day values which are common are scaled back out. We default this to 30.473 which is an often used approximation for a month.
    """

    pre: EnergyChangepointModelRequest = Field(
        ..., description="The pre-retrofit data for this option-c request."
    )
    post: EnergyChangepointModelRequest = Field(
        ..., description="The post-retrofit data for this option-c request."
    )
    confidence_interval: Optional[ConfidenceInterval] = Field(
        0.8,
        description="Optional confidence interval for savings calculations using monthly data. The library default value is 80%",
    )
    scalar: Optional[Scalar] = Field(
        30.473,
        description="Value to scale savings data. If giving data by per/day values, the default of 30.473 will scale these values to a correct monthly value.",
    )
    norms: Optional[List[float]] = Field(
        None,
        description="Optional normalized outside air temperature data. Used to calculate NAC. Length must be 12 (monthly)",
    )

    @pydantic.validator("pre", "post")
    def validate_usage(cls, v: Any) -> Any:
        if len(v.usage.usage) != 12 or len(v.usage.oat) != 12:
            raise ValueError(
                "Pre and post number of points must be 12 to perform monthly savings calculations."
            )
        return v

    @pydantic.validator("norms")
    def validate_norms(cls, v: Any) -> Any:
        if v and len(v) != 12:
            raise ValueError(
                "Invalid length of normalized temperature data. Length must be 12 to perform monthly savings calculations."
            )
        return v


class PredictedSum(pydantic.BaseModel):
    """The return value calculated for normalized annual consumption."""

    value: float

    def parse(self) -> Dict[str, Union[float, str]]:
        """Make this one into nac

        Returns:
            Dict[str, Union[float, str]]: _description_
        """
        return {"measurementtype": "normalized_annual_consumption", "value": self.value}


class Load(pydantic.BaseModel):
    """Base heating and cooling loads for each model."""

    base: float
    heating: float
    cooling: float

    def parse(self) -> List[Dict[str, Union[float, str]]]:
        """Simple business logic to create loads records for cpmmodelmeasurement that
        can go back into bemadb.

        Returns:
            List[Dict[str, Union[float, str]]]: _description_
        """
        return [
            {"measurementtype": f"{k}_load", "value": v} for k, v in self.dict().items()
        ]


class EnergyParameterModelCoefficients(pydantic.BaseModel):
    """The model parameters determined from curve_fit. Since 2P models do not have changepoints this
    value may be None.
    """

    yint: float
    slopes: List[float]
    changepoints: Optional[List[float]]

    def parse(self, model_type: str) -> GenericRecords:
        """Complicated business logic to parse and flatten changepointmodel coefficients to
        cpmodelmeasurements so they can go into bemadb for lean rank.

        Args:
            model_type (str): The name of the model_type provided by caller.

        Raises:
            AssertionError: _description_

        Returns:
            _type_: _description_
        """
        if model_type == "2P":
            assert len(self.slopes) == 1
            if self.slopes[0] < 0:
                return [
                    {"measurementtype": "heating_sensitivity", "value": self.slopes[0]}
                ]
            else:
                return [
                    {"measurementtype": "cooling_sensitivity", "value": self.slopes[0]}
                ]

        elif model_type == "3PC":
            assert len(self.slopes) == 1
            assert self.changepoints is not None and len(self.changepoints) == 1
            return [
                {
                    "measurementtype": "cooling_sensitivity",  # NOTE that I am not checking slope here since this should always be positive based on maths in changepointmodel!
                    "value": self.slopes[0],
                },
                {
                    "measurementtype": "cooling_changepoint",
                    "value": self.changepoints[0],
                },
            ]
        elif model_type == "3PH":
            assert len(self.slopes) == 1
            assert self.changepoints is not None and len(self.changepoints) == 1
            return [
                {
                    "measurementtype": "heating_sensitivity",  # NOTE that I am not checking slope here since this should always be negative based on maths in changepointmodel!
                    "value": self.slopes[0],
                },
                {
                    "measurementtype": "heating_changepoint",
                    "value": self.changepoints[0],
                },
            ]
        elif model_type == "4P":
            assert len(self.slopes) == 2
            assert self.changepoints is not None and len(self.changepoints) == 1
            return [
                {"measurementtype": "heating_sensitivity", "value": self.slopes[0]},
                {"measurementtype": "cooling_sensitivity", "value": self.slopes[1]},
                {
                    "measurementtype": "heating_changepoint",  # NOTE same cp for both
                    "value": self.changepoints[0],
                },
                {
                    "measurementtype": "cooling_changepoint",  # NOTE same cp for both
                    "value": self.changepoints[0],
                },
            ]
        elif model_type == "5P":
            assert len(self.slopes) == 2
            assert self.changepoints is not None and len(self.changepoints) == 2
            return [
                {"measurementtype": "heating_sensitivity", "value": self.slopes[0]},
                {"measurementtype": "cooling_sensitivity", "value": self.slopes[1]},
                {
                    "measurementtype": "heating_changepoint",
                    "value": self.changepoints[0],
                },
                {
                    "measurementtype": "cooling_changepoint",
                    "value": self.changepoints[1],
                },
            ]

        else:
            raise AssertionError("Should never get here :/")


class Score(pydantic.BaseModel):
    """Holds onto the score values for r2 and cvrmse."""

    name: str
    value: float
    threshold: float
    ok: bool

    def parse(self) -> Tuple[str, float]:
        """We need to expose the name and value for bemadb.

        Returns:
            Tuple[str, float]: Name and value
        """

        return self.name, self.value

    def parse_for_csv(self) -> Dict[str, Any]:
        return {
            self.name: self.value,
            f"{self.name}_threshold": self.threshold,
            f"{self.name}_ok": self.ok,
        }


class EnergyChangepointModelInputData(pydantic.BaseModel):
    """We return the original input data in a more stripped down manner."""

    X: OneDimNDArrayField
    y: OneDimNDArrayField

    def get_Xy_as_oat_usage(self) -> Dict[str, List[float]]:
        return {"oat": self.X.tolist(), "usage": self.y.tolist()}


class EnergyChangepointModelResult(pydantic.BaseModel):
    """A result object for a single changepointmodel."""

    name: str
    coeffs: EnergyParameterModelCoefficients
    pred_y: OneDimNDArrayField
    load: Load
    scores: List[Score]
    input_data: EnergyChangepointModelInputData
    nac: Optional[PredictedSum] = None

    class Config(NpConfig):
        ...

    def _parse_scores_for_csv(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for score in self.scores:
            res = score.parse_for_csv()
            out = {**out, **res}
        return out

    def _flatten_cp_model_measurements(self) -> Dict[str, Any]:
        res = self.get_cpmodelmeasurements()
        _measurements = [
            "heating_sensitivity",
            "cooling_sensitivity",
            "heating_changepoint",
            "cooling_changepoint",
            "normalized_annual_consumption",
        ]
        act_measurement = {rec["measurementtype"]: rec["value"] for rec in res}
        none_measurement = {
            m_type: None
            for m_type in _measurements
            if m_type not in act_measurement.keys()
        }
        return {**act_measurement, **none_measurement}  # type: ignore

    def assemble_full_cp_model_result(self, prepost: str = "") -> Dict[str, Any]:
        """assemble rpc cp model result into flat structure composed of cp model results: socres, coefficients/measurements,
            model_type etc.; this is for csv parsing.

        Args:
            prepost (str, optional): Add prepost key value label to cp model result if provided. Defaults to ''.

        Returns:
            Dict[str, Any]: a flat dict representation of rpc cp model result.
        """
        measurements = self._flatten_cp_model_measurements()
        scores = self._parse_scores_for_csv()
        prepost_ = {"prepost": prepost} if prepost else {}
        return {**measurements, **scores, **prepost_, "model_type": self.name}

    def get_pred_y(self, prepost: str = "") -> GenericRecords:
        """generate a list of predicted usage from rpc cp model result; for csv parsing.

        Args:
            prepost (str, optional): add prepost key value label to records if provided. Defaults to ''.

        Returns:
            GenericRecords: a list of predicted usage records with model_type and, optionally, prepost info.
        """
        meta_data = (
            {"prepost": prepost, "model_type": self.name}
            if prepost
            else {"model_type": self.name}
        )
        return [{"predicted_usage": i, **meta_data} for i in self.pred_y]

    def get_cpmodelmeasurements(
        self, add_model_type: bool = False
    ) -> List[Dict[str, Union[float, str]]]:
        """Parse and merge all valid cpmodelmeasurements for this result.
            Can be used for writing output into bemadb changepointmodelmeasurement table. For writing output
            directly to db, set add_model_type to False.

        Args:
            add_model_type (bool, optional): Add model type to measurements result. Defaults to False.

        Returns:
            List[Dict[str, Union[float, str]]]: dict representation of changepointmodelmeasurement db schemas
        """
        out = []
        out.extend(self.load.parse())  # loads
        out.extend(self.coeffs.parse(self.name))  # cps + sensitivities
        if self.nac is not None:  # just in case we don't have nac for some reason...
            out.append(self.nac.parse())
        if add_model_type:  # add modeltype for the most public
            return [{**i, "model_type": self.name} for i in out]
        return out

    def parse_input_data(self) -> Dict[str, List[float]]:
        """Parse the InputData object as oat/usage.

        Returns:
            Dict[str, List[float]]: The oat/usage pair.
        """
        return self.input_data.get_Xy_as_oat_usage()

    def get_scores(self) -> Dict[str, float]:  # {r2: 42, cvrmse: 42}
        """Needed for top level changepointmodel result.

        Returns:
            Dict[str, float]: Scores
        """
        out = {}
        for score in self.scores:
            name, val = score.parse()
            out[name] = val
        return out

    def make_cp_model_result(self) -> Dict[str, Any]:
        """method to make cpmodel results for bemadb changepointmodelresult table.

        Returns:
            _type_: dict representation of changepointmodelresult db schemas
        """
        cpmodel = self.dict(exclude={"input_data", "pred_y"})
        cpmodel[
            "pred_y"
        ] = (
            self.pred_y.tolist()
        )  # manual conversion to list here since it may not be properly serialized to JSON in some cases
        cpmodel["input_data"] = self.parse_input_data()
        return {
            **self.get_scores(),
            "modeltype": self.name,
            "result": {"cpmodel": cpmodel},
        }


class AdjustedSavingsResult(pydantic.BaseModel):
    adjusted_y: OneDimNDArrayField
    total_savings: float
    average_savings: float
    percent_savings: float
    percent_savings_uncertainty: float

    def parse_saving_results(
        self, pre_model_type: str, post_model_type: str
    ) -> Dict[str, Any]:
        """parse adjusted saving result to a flat representation; for csv parsing.

        Args:
            pre_model_type (str): pre model type name
            post_model_type (str): post model type name

        Returns:
            Dict[str, Any]: adjusted saving result dict
        """
        result = self.dict(exclude={"adjusted_y"})
        return {
            "savings_type": "adjusted",
            "pre_model_type": pre_model_type,
            "post_model_type": post_model_type,
            **result,
        }

    def parse_adjusted_usage(self, prepost: str) -> GenericRecords:
        """generate a list of adjusted usage records; for csv parsing.

        Args:
            prepost (str): add prepost label to adjusted usage records.

        Returns:
            List[Dict[str,Any]]: a list of adjusted usage records
        """
        if prepost == "pre":
            return [
                {"adjusted_usage": None, "prepost": "pre"}
                for i in range(len(self.adjusted_y))
            ]
        else:
            return [{"adjusted_usage": i, "prepost": "post"} for i in self.adjusted_y]


class NormalizedSavingsResult(pydantic.BaseModel):
    normalized_y_pre: OneDimNDArrayField
    normalized_y_post: OneDimNDArrayField
    total_savings: float
    average_savings: float
    percent_savings: float
    percent_savings_uncertainty: float

    def parse_saving_results(
        self, pre_model_type: str, post_model_type: str
    ) -> Dict[str, Any]:
        """parse normalized saving result to a flat representation; for csv parsing.

        Args:
            pre_model_type (str): pre model type name
            post_model_type (str): post model type name

        Returns:
            Dict[str, Any]: normalized saving result dict
        """
        result = self.dict(exclude={"normalized_y_post", "normalized_y_pre"})
        return {
            "savings_type": "normalized",
            "pre_model_type": pre_model_type,
            "post_model_type": post_model_type,
            **result,
        }

    def parse_normalized_usage(
        self, prepost: str, model_name: str = ""
    ) -> GenericRecords:
        """generate a list of nromalized usage records; for csv parsing.

        Args:
            prepost (str): add prepost label to normalized usage records.
            model_name (str): add model type name to normalized usage records

        Returns:
            List[Dict[str,Any]]: a list of normalized usage records
        """
        data = self.normalized_y_pre if prepost == "pre" else self.normalized_y_post
        model = {"model_type": model_name} if model_name else {}
        return [
            {"prepost": prepost, "normalized_usage": rec, "month": i, **model}
            for i, rec in enumerate(data, start=1)
        ]


class AdjustedSavingsResultData(pydantic.BaseModel):
    """An adjusted savings result."""

    confidence_interval: float
    result: AdjustedSavingsResult

    class Config(NpConfig):
        ...


class NormalizedSavingsResultData(pydantic.BaseModel):
    """A normalized savings result"""

    confidence_interval: float
    result: NormalizedSavingsResult

    class Config(NpConfig):
        ...


class SavingsResult(pydantic.BaseModel):
    """A single savings result object."""

    pre: EnergyChangepointModelResult
    post: EnergyChangepointModelResult
    adjusted_savings: AdjustedSavingsResultData
    normalized_savings: Optional[NormalizedSavingsResultData] = None

    class Config(NpConfig):
        ...

    def parse_prepost_cp_model_results(
        self,
    ) -> GenericRecords:
        """parse pre and post rpc cp model from savings endpoint to a list of flat pre and post cp models

        Returns:
            GenericRecords: list of cp models.
        """
        pre_model = self._parse_cp_models(prepost="pre")
        post_model = self._parse_cp_models(prepost="post")
        return [pre_model, post_model]

    def _parse_cp_models(self, prepost: str) -> Dict[str, Any]:
        result = self.pre if prepost == "pre" else self.post
        return result.assemble_full_cp_model_result(prepost=prepost)

    def get_predicted_usage(self, prepost: str) -> GenericRecords:
        """generate a list of predicted usage; for csv parsing.

        Args:
            prepost (str): prepost label

        Returns:
            GenericRecords: a list of predicted usage
        """
        result = self.pre if prepost == "pre" else self.post
        return result.get_pred_y(prepost=prepost)

    def parse_savings_result(
        self,
    ) -> GenericRecords:
        """parse adjusted and normalized savings data from savings endpoint to a list
        of flat and structured saving results.

        Returns:
            GenericRecords: saving results such as percent savings, average savings and percent saving uncertainty.
        """
        pre_model_type = self._get_model_type(prepost="pre")
        post_model_type = self._get_model_type(prepost="post")
        adj = self.adjusted_savings.result.parse_saving_results(
            pre_model_type, post_model_type
        )
        norm = (
            self.normalized_savings.result.parse_saving_results(
                pre_model_type, post_model_type
            )
            if self.normalized_savings
            else {}
        )
        return [adj, norm]

    def parse_saving_usage(
        self,
    ) -> Tuple[Any, Any]:
        """parse adjusted and normalized savings data from savings endpoint to a list
        of flat and structured saving related calculation usage.

        Returns:
            Tuple[GenericRecords]: saving related calculation usage such as adjusted and normalized usage.
        """
        pre_savinge_usage, pre_norm = self._gather_savings_related_usage(prepost="pre")
        post_savinge_usage, post_norm = self._gather_savings_related_usage(
            prepost="post"
        )
        return pre_savinge_usage + post_savinge_usage, pre_norm + post_norm

    def _get_model_type(self, prepost: str) -> str:
        model = self.pre if prepost == "pre" else self.post
        return model.name

    def _gather_savings_related_usage(
        self, prepost: str
    ) -> Tuple[GenericRecords, GenericRecords]:
        model_name = self._get_model_type(prepost=prepost)
        predicted = self.get_predicted_usage(
            prepost=prepost
        )  # prepost and model name included
        adj = self.adjusted_savings.result.parse_adjusted_usage(prepost=prepost)
        assert self.normalized_savings is not None
        norm = self.normalized_savings.result.parse_normalized_usage(
            prepost=prepost, model_name=model_name
        )
        return [
            {**predicted_, **adj_} for predicted_, adj_ in zip(adj, predicted)
        ], norm


class EnergyChangepointModelResponse(pydantic.BaseModel):
    """A list of changepointmodel results."""

    results: List[EnergyChangepointModelResult]

    class Config(NpConfig):
        ...

    def parse_cp_measurements(self) -> GenericRecords:
        """parse rpc changepoint model results list to prodcue cp measurements form for bemadb"""
        out = []
        for res in self.results:
            out.extend(res.get_cpmodelmeasurements(add_model_type=True))
        return out

    def parse_results_for_csv(self) -> Tuple[GenericRecords, GenericRecords]:
        """parse rpc changepoint model results list to produce cp model lists and predicted usage list;
        this is for csv parsing.

        Returns:
            Tuple[GenericRecords]: tuple of length 4; cp model lists, saving result list,
                                            adjusted usage and normalized usage.
        """
        model_output = []
        pred_y_output = []
        for res in self.results:
            model_output.append(res.assemble_full_cp_model_result(prepost="pre"))
            pred_y_output.extend(res.get_pred_y(prepost="pre"))
        return model_output, pred_y_output


class SavingsResponse(pydantic.BaseModel):
    """A list of savings results."""

    results: List[SavingsResult]

    class Config(NpConfig):
        ...

    def parse_results_for_csv(
        self,
    ) -> Tuple[GenericRecords, GenericRecords, GenericRecords, GenericRecords]:
        """parse rpc savings model results list to prodcue cp model lists, saving result list,
            adjusted usage and normalized usage; for csv parsing.

        Returns:
            Tuple[GenericRecords, GenericRecords, GenericRecords, GenericRecords]:
                model output, savings result, savings related usage and norms output
        """
        model_output: GenericRecords = []  # both pre and post up to 25
        savings_results_output: GenericRecords = []
        savings_related_usage_output: GenericRecords = []
        norms_output: GenericRecords = []
        for res in self.results:
            model_output.extend(res.parse_prepost_cp_model_results())
            savings_results_output.extend(res.parse_savings_result())
            u, n = res.parse_saving_usage()
            savings_related_usage_output.extend(u)
            norms_output.extend(n)
        return (
            model_output,
            savings_results_output,
            savings_related_usage_output,
            norms_output,
        )
