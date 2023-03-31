from typing import Tuple
import numpy as np
from . import uncertainties


def adjusted(
    gross_adjusted_pred_y: float,
    gross_post_y: float,
    pre_cvrmse: float,
    pre_p: int,
    pre_n: int,
    post_n: int,
    confidence_interval: float = 0.8,
    scalar: float = None,
) -> Tuple[float, float, float, float]:
    """The adjusted savings uncertainty calculation for option-c methodology as defined in ashrae and implemented by BPL.

    Args:
        gross_adjusted_pred_y (float): The gross adjusted value predicted y.
        gross_post_y (float): The gross y value for the post retrofit model.
        pre_cvrmse (float): The cvrmse of the pre retrofit model.
        pre_p (int): The number of parameters in the pre retrofit model.
        pre_n (int): The number of points used to fit the pre retrofit model.
        post_n (int): The number of points userd to the fit the post retrofit model. NOTE must equal the pre_n.
        confidence_interval (float, optional): The confidence. Defaults to 0.8.
        scalar (float, optional): Value to scale by. Use 30.437 to scale from per-day to total month!!

    Returns:
        Tuple[float, float, float, float]: A tuple of total_savings, average_savings, percent_savings and percent_savings_uncertainity.
    """
    assert pre_n == post_n, "pre_n and post_n must be equal"

    scalar = 1 if scalar is None else scalar
    total_savings = (gross_adjusted_pred_y - gross_post_y) * scalar
    percent_savings = total_savings / gross_post_y

    fractional_savings = np.absolute(
        uncertainties.fractional_avoided_energy_use(gross_adjusted_pred_y, gross_post_y)
    )

    rel_unc = uncertainties.relative_uncertainty_avoided_energy_use(
        confidence_interval, pre_cvrmse, fractional_savings, pre_p, pre_n, post_n
    )

    absolute_uncertainty_of_total_savings = rel_unc * np.absolute(total_savings)

    average_savings = total_savings / post_n
    percent_savings_uncertainty = absolute_uncertainty_of_total_savings / np.absolute(
        total_savings
    )

    return total_savings, average_savings, percent_savings, percent_savings_uncertainty


def weather_normalized(
    gross_normalized_pred_y_pre: float,
    gross_normalized_pred_y_post: float,
    pre_cvrmse: float,
    post_cvrmse: float,
    pre_n: int,
    post_n: int,
    pre_p: int,
    post_p: int,
    n_norm: int,
    confidence_interval: float = 0.8,
    scalar: float = None,
) -> Tuple[float, float, float, float]:
    """The weather normalized savings uncertainty calculation for option-c methodology as defined in ashrae and implemented by BPL.
    Will check that pre_n post_n and n_norm are equal before proceeding with the calculation.

    Args:
        gross_normalized_y_pre (float): The gross normalized predicted y value for pre retrofit model.
        gross_normalized_y_post (float): The gross normlized predicted y value for post retrofit model.
        pre_cvrmse (float): The cvrmse of the pre retrofit model.
        post_cvrmse (float): The cvrmse of the post retrofit model.
        pre_n (int): The number of points in the pre model. Must equal post_n and n_norm.
        post_n (int): The number of points in the post model. Must equ pre_n and n_norm.
        pre_p (int): The number of parameters in the pre model.
        post_p (int): The number of parameters in the post model.
        n_norm (int): The number of parameters in the noramlized model. Must equal pre_n and post_n.
        confidence_interval (float, optional): The confidence interval of the uncertainity calculation. Defaults to 0.8.
        scalar (float, optional): Value to scale by. Use 30.437 to scale from per-day to total month!!

    Returns:
        Tuple[float, float, float, float]: A tuple of total_savings, average_savings, percent_savings and percent_savings_uncertainity.
    """

    assert pre_n == post_n == n_norm, "pre_n, post_n, and n_norm must be the same"

    scalar = 1 if scalar is None else scalar
    total_savings = (
        gross_normalized_pred_y_pre - gross_normalized_pred_y_post
    ) * scalar
    percent_savings = total_savings / gross_normalized_pred_y_pre

    pre_rel_unc = uncertainties.relative_uncertainty_normalized_period(
        confidence_interval,
        pre_n,
        pre_cvrmse,
        gross_normalized_pred_y_pre,
        pre_p,
        n_norm,
    )

    post_rel_unc = uncertainties.relative_uncertainty_normalized_period(
        confidence_interval,
        post_n,
        post_cvrmse,
        gross_normalized_pred_y_post,
        post_p,
        n_norm,
    )

    absolute_savings_uncertainty = np.sqrt((pre_rel_unc) ** 2 + (post_rel_unc) ** 2)
    percent_savings_uncertainty = np.absolute(
        absolute_savings_uncertainty / total_savings
    )
    average_savings = total_savings / n_norm

    return total_savings, average_savings, percent_savings, percent_savings_uncertainty
