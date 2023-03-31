""" ashrae formulas for option-c methodologies involving energy usage.
"""
import numpy as np
from scipy import stats
from ..nptypes import OneDimNDArray


def _get_tstat(confidence_interval: float, df: int) -> float:
    return stats.t.interval(confidence_interval, df)[1]


def fractional_avoided_energy_use(
    total_annual_adjusted: float, total_annual_measured_reporting: float
) -> float:
    """ASHRAE Guideline 14 2014 Annex B Eq. B-22 Fractional Avoided Energy Use

    Args:
        annual_adj_baseline (float): Sum of a year of adjusted baseline consumption.
        annual_measured_reporting (float): Sum of a year of measured reporting consumption.

    Returns:
        float: Savings as a fraction of annual adjusted baseline.
    """
    return (
        total_annual_adjusted - total_annual_measured_reporting
    ) / total_annual_adjusted


def relative_uncertainty_avoided_energy_use(
    confidence_interval: float, cvrmse: float, f: float, p: int, n_pre: int, n_post: int
) -> float:
    """
     ASHARE Guideline 14 2014, Annex B
     Modified Eq. B-28
     Relative Uncertainty of Avoided Energy Use

    Args:
        conf_interval (float): desired confidence interval
        cvrmse (float): cvrmse of baseline model as a fraction
        f (float): savings as a fraction of annual adjusted baseline
        p (int): number of model parameters. depends on model complexity and number of independent variables.
        len_pre (int): number of points pre
        len_post (int): number of points post

    Returns:
        float: The relative uncertainties of avoided energy use
    """

    tstat = _get_tstat(confidence_interval, n_pre - p)
    return 1.26 * tstat * cvrmse * np.sqrt((1 / n_post) * (1 + 2 / n_pre)) / f


def relative_uncertainty_normalized_period(
    confidence_interval: float,
    n: int,
    cvrmse: float,
    gross_norm: float,
    p: int,
    n_norm: int,
) -> float:
    """
     ASHARE Guideline 14 2014, Annex B
     Modified Eq. B-28
     Relative Uncertainty of Normalized Period
     This function is used twice in normalized savings uncertainty calculation.

    Args:
        conf_interval (float): desired confidence interval
        n (int): number of points of data used to create the model
        cvrmse (float): model cvrmse as a fraction
        gross_norm (float): sum of normalized consumption for the number of cny months
        p (int): number of parameters. Depends on model complexity and number of independent variables.
        n_norm (int): number of points in normal period

    Returns:
        float: uncertainty for sum of normalized consumption for that period
    """

    tstat = _get_tstat(confidence_interval, n - p)
    return 1.26 * tstat * gross_norm * cvrmse * np.sqrt((1 / n_norm) * (1 + 2 / n))
